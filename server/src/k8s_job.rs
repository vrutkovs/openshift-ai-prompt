use crate::{ws, Message};
use futures_util::{stream::SplitSink, SinkExt};
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

#[derive(Clone, Envconfig)]
pub struct JobSettings {
    #[envconfig(from = "GENERATE_IMAGE")]
    pub image: String,
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
    prompt: String,
    ws_sender: &mut SplitSink<WebSocketStream<TcpStream>, Message>,
) -> Result<String, anyhow::Error> {
    ws_sender.send(ws::status_message("Starting")).await?;

    let s3_settings = S3Settings::init_from_env()?;
    let job_settings = JobSettings::init_from_env()?;

    let s: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect();
    let result_filename = format!("{s}.png");

    let client = Client::try_default().await?;
    let jobs: Api<Job> = Api::default_namespaced(client);
    let job_json = create_job_for_prompt(
        prompt,
        job_settings.image,
        s3_settings.clone(),
        result_filename.clone(),
    )?;
    let data = jobs.create(&PostParams::default(), &job_json).await?;
    let job_name = data.name_any();
    ws_sender
        .send(ws::status_message(
            format!("Created job {job_name}").as_str(),
        ))
        .await?;

    let cond = await_condition(jobs.clone(), &job_name, conditions::is_job_completed());
    let _ =
        tokio::time::timeout(std::time::Duration::from_secs(job_settings.timeout), cond).await?;
    jobs.delete(&job_name, &DeleteParams::background()).await?;

    ws_sender.send(ws::status_message("Job completed")).await?;
    let bucket = s3_settings.s3_bucket;
    Ok(format!(
        "https://{bucket}.s3.amazonaws.com/{result_filename}"
    ))
}

pub fn create_job_for_prompt(
    prompt: String,
    image: String,
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
                },
                "spec": {
                    "containers": [{
                        "name": "generate",
                        "image": image,
                        "env": {
                            "PROMPT": prompt,
                            "RESULT_FILENAME": result_filename,
                            "AWS_ACCESS_KEY": s3_settings.s3_key,
                            "AWS_ACCESS_SECRET": s3_settings.s3_secret,
                            "S3_BUCKET_NAME": s3_settings.s3_bucket,
                        },
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": 1,
                            }
                        }
                    }],
                    "restartPolicy": "Never",
                }
            }
        }
    }))
}
