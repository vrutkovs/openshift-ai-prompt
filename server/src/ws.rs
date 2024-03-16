use anyhow::Error;
use serde::Serialize;
use tokio_tungstenite::tungstenite::Message;

#[derive(Default, Serialize)]
pub enum WSMessageType {
    #[default]
    Progress,
    Result,
    Error,
}

#[derive(Default, Serialize)]
pub struct WSMessage {
    #[serde(rename(serialize = "type", deserialize = "type"))]
    msgtype: WSMessageType,
    message: Option<String>,
}

pub trait AsWS {
    fn as_ws(&self) -> Message;
}

impl AsWS for WSMessage {
    fn as_ws(&self) -> Message {
        match serde_json::to_string(&self) {
            Ok(j) => Message::Text(j.to_owned()),
            Err(e) => Message::Text(e.to_string()),
        }
    }
}

pub fn progress(status: &str) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Progress,
        message: Some(status.to_string()),
    }
}

pub fn result(url: String) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Result,
        message: Some(url),
    }
}

pub fn error(error: Error) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Error,
        message: Some(error.to_string()),
    }
}
