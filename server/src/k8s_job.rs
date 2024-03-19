use crate::Message;
use log::*;

use anyhow::Context;
use futures_util::{stream::SplitSink, SinkExt};
use openshift_ai_prompt_common::ws::{self, WSMessage};
use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;

use k8s_openapi::api::batch::v1::Job;
use kube::{
    api::{Api, DeleteParams, PostParams},
    runtime::wait::{await_condition, conditions},
    Client, ResourceExt,
};

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

    info!(
        "Peer {}: generating '{}' and saving to {}",
        peer, prompt, result_filename
    );

    let client = Client::try_default()
        .await
        .context("Failed to create k8s client")?;
    let jobs: Api<Job> = Api::default_namespaced(client);
    let job_json = create_job_for_prompt(
        prompt,
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

    let cond = await_condition(jobs.clone(), &job_name, conditions::is_job_completed());
    let _ = tokio::time::timeout(std::time::Duration::from_secs(job_settings.timeout), cond)
        .await
        .context("Timeout waiting for job to complete")?;
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

pub fn create_job_for_prompt(
    prompt: String,
    job_settings: JobSettings,
    s3_settings: S3Settings,
    result_filename: String,
) -> Result<Job, serde_json::Error> {
    serde_json::from_value(serde_json::json!({
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": "picture-",
        },
        "spec": {
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
                        "env": [
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
                            }
                        ],
                        "imagePullPolicy": "Always",
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": "1",
                            }
                        },
                        "volumeMounts": [{
                            "name": "model",
                            "mountPath": "/opt/app-root/src/model",
                            "subPath": "model",
                        }]
                    }],
                    "volumes": [{
                        "name": "model",
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
