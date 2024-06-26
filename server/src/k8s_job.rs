use crate::models::Join;
use crate::{models, Message};
use log::*;

use crate::k8s_job::conditions::Condition;
use anyhow::{Context, Error};
use futures_util::TryStreamExt;
use futures_util::{stream::SplitSink, AsyncBufReadExt, SinkExt};
use k8s_openapi::api::batch::v1::Job;
use k8s_openapi::api::core::v1::Pod;
use kube::{
    api::{Api, DeleteParams, ListParams, LogParams, PostParams},
    runtime::wait::{await_condition, conditions},
    Client, ResourceExt,
};
use openshift_ai_prompt_common::ws::{self, WSMessage};
use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;

use envconfig::Envconfig;
use rand::{distributions::Alphanumeric, Rng};

use tokio_tungstenite::tungstenite::Message as tungstenite_Message;

pub trait AsTungstenite {
    fn as_msg(&self) -> tungstenite_Message;
}

impl AsTungstenite for WSMessage {
    fn as_msg(&self) -> tungstenite_Message {
        match serde_json::to_string(&self) {
            Ok(j) => tungstenite_Message::Text(j.to_owned()),
            Err(e) => {
                let backtrace = std::backtrace::Backtrace::capture();
                tungstenite_Message::Text(format!("{e}: {:#?}", backtrace))
            }
        }
    }
}

pub trait AsWSMessage {
    fn as_msg(&self) -> Result<WSMessage, Error>;
}

impl AsWSMessage for tungstenite_Message {
    fn as_msg(&self) -> Result<WSMessage, Error> {
        if !self.is_text() {
            anyhow::bail!("not text");
        }
        Ok(serde_json::from_str(&self.to_string())?)
    }
}

#[derive(Clone, Envconfig)]
pub struct JobSettings {
    #[envconfig(from = "GENERATE_IMAGE")]
    pub image: String,
    #[envconfig(from = "GENERATE_DEVICE")]
    pub device: String,
    #[envconfig(from = "GENERATE_STEPS")]
    pub steps: u64,
    #[envconfig(from = "GENERATE_TIMEOUT")]
    pub timeout: u64,
}

#[derive(Clone, Envconfig)]
pub struct S3Settings {
    #[envconfig(from = "AWS_ACCESS_SECRET")]
    pub s3_secret: String,
    #[envconfig(from = "AWS_ACCESS_KEY")]
    pub s3_key: String,
    #[envconfig(from = "S3_BUCKET_NAME")]
    pub s3_bucket: String,
}

pub async fn start(
    peer: String,
    prompt: String,
    model: models::Model,
    ws_sender: &mut SplitSink<WebSocketStream<TcpStream>, Message>,
) -> Result<String, anyhow::Error> {
    ws_sender
        .send(ws::progress("Request received", 0.1).as_msg())
        .await?;

    let s3_settings = S3Settings::init_from_env().context("S3 settings")?;
    let job_settings = JobSettings::init_from_env().context("Job settings")?;

    let s: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect();
    let result_filename = format!("{s}.png");

    let amended_prompt = match model.triggers() {
        Some(s) => format!("{} {}", prompt, s),
        None => prompt,
    };
    info!(
        "Peer {}: generating '{}' and saving to {}",
        peer, amended_prompt, result_filename
    );

    let client = Client::try_default()
        .await
        .context("Failed to create k8s client")?;
    let jobs: Api<Job> = Api::default_namespaced(client.clone());
    let job_json = create_job_for_prompt(
        amended_prompt,
        model,
        job_settings.clone(),
        s3_settings.clone(),
        result_filename.clone(),
    )
    .context("Job cannot be created")?;
    let data = jobs
        .create(&PostParams::default(), &job_json)
        .await
        .context("Job cannot be applied")?;
    info!("Peer {}: created job '{}'", peer, data.name_any());

    let job_name = data.name_any();
    ws_sender
        .send(ws::progress(format!("Created job {job_name}").as_str(), 0.5).as_msg())
        .await
        .context("Failed to send progress message")?;

    let cond = await_condition(jobs.clone(), &job_name, job_has_running_pods());
    let _ = tokio::time::timeout(std::time::Duration::from_secs(job_settings.timeout), cond)
        .await
        .context("Timeout waiting for job to start pods")?;

    let pod_label_selector = format!("job-name={}", job_name);
    let pods: Api<Pod> = Api::default_namespaced(client);
    let pod_list = pods
        .list(&ListParams::default().labels(pod_label_selector.as_str()))
        .await?;
    let pod = pod_list
        .iter()
        .last()
        .context("Unable to find started pods")?;
    let pod_name = pod.name_any();
    ws_sender
        .send(ws::progress(format!("Created pod {pod_name}").as_str(), 0.7).as_msg())
        .await
        .context("Failed to send progress message")?;

    let cond = await_condition(pods.clone(), &pod_name, pod_is_started());
    let _ = tokio::time::timeout(std::time::Duration::from_secs(job_settings.timeout), cond)
        .await
        .context("Timeout waiting for pod to become running")?;

    let mut logs = pods
        .log_stream(
            &pod_name,
            &LogParams {
                follow: true,
                pretty: true,
                ..LogParams::default()
            },
        )
        .await?
        .lines();
    while let Some(line) = logs.try_next().await? {
        ws_sender
            .send(ws::progress(line.as_str(), 0.8).as_msg())
            .await
            .context("Failed to send progress message")?;
    }

    let cond = await_condition(jobs.clone(), &job_name, job_succeeded());
    let _ = tokio::time::timeout(std::time::Duration::from_secs(job_settings.timeout), cond)
        .await
        .context("Timeout waiting for job status check")?;

    jobs.delete(&job_name, &DeleteParams::background())
        .await
        .context("Failed to delete job")?;
    ws_sender
        .send(ws::progress("Job completed", 0.9).as_msg())
        .await?;
    info!("Peer {}: job {} completed", peer, data.name_any());

    let bucket = s3_settings.s3_bucket;
    Ok(format!(
        "https://{bucket}.s3.amazonaws.com/{result_filename}"
    ))
}

pub fn job_succeeded() -> impl Condition<Job> {
    |obj: Option<&Job>| {
        if let Some(job) = &obj {
            if let Some(s) = &job.status {
                if let Some(succeeded) = &s.succeeded {
                    return succeeded > &0;
                }
            }
        }
        false
    }
}

pub fn job_has_running_pods() -> impl Condition<Job> {
    |obj: Option<&Job>| {
        if let Some(job) = &obj {
            if let Some(s) = &job.status {
                if let Some(active) = &s.active {
                    return active > &0;
                }
            }
        }
        false
    }
}

pub fn pod_is_started() -> impl Condition<Pod> {
    |obj: Option<&Pod>| {
        if let Some(pod) = &obj {
            if let Some(s) = &pod.status {
                if let Some(phase) = &s.phase {
                    return phase == "Running";
                }
            }
        }
        false
    }
}

pub fn create_job_for_prompt(
    prompt: String,
    model: models::Model,
    job_settings: JobSettings,
    s3_settings: S3Settings,
    result_filename: String,
) -> Result<Job, serde_json::Error> {
    let base_model = model.base_model().unwrap_or(model);
    let env = serde_json::json!([
        {
            "name": "PROMPT",
            "value": prompt,
        }, {
            "name": "RESULT_FILENAME",
            "value": result_filename,
        }, {
            "name": "AWS_ACCESS_KEY",
            "value": s3_settings.s3_key,
        }, {
            "name": "AWS_ACCESS_SECRET",
            "value": s3_settings.s3_secret,
        }, {
            "name": "S3_BUCKET_NAME",
            "value": s3_settings.s3_bucket,
        }, {
            "name": "OPENJOURNEY_DEVICE",
            "value": job_settings.device,
        }, {
            "name": "OPENJOURNEY_STEPS",
            "value": job_settings.steps.to_string(),
        }, {
            "name": "OPENJOURNEY_MODEL",
            "value": base_model.subpath(),
        }, {
            "name": "ADAPTER_NAMES",
            "value": model.adapters().names(),
        }, {
            "name": "ADAPTER_PATHS",
            "value": model.adapters().paths(),
        }, {
            "name": "ADAPTER_WEIGHTS",
            "value": model.adapters().weights(),
        }, {
            "name": "PYTHONUNBUFFERED",
            "value": "1",
        }, {
            "name": "TQDM_POSITION",
            "value": "1",
        }
    ]);

    serde_json::from_value(serde_json::json!({
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": "picture-",
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "generateName": "picture-",
                    "annotations": {
                        "alpha.image.policy.openshift.io/resolve-names": "'*'",
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "generate",
                        "image": job_settings.image,
                        "env": env,
                        "imagePullPolicy": "Always",
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": "1",
                            }
                        },
                        "volumeMounts": [{
                            "name": "models",
                            "mountPath": "/opt/app-root/src/models",
                        }]
                    }],
                    "volumes": [{
                        "name": "models",
                        "persistentVolumeClaim": {
                            "claimName": "model",
                        }
                    }],
                    "restartPolicy": "Never",
                }
            }
        }
    }))
}
