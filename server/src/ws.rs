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
    msgtype: WSMessageType,
    status: Option<String>,
    url: Option<String>,
    error: Option<String>,
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
        status: Some(status.to_string()),
        ..Default::default()
    }
}

pub fn result(url: String) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Result,
        url: Some(url),
        ..Default::default()
    }
}

pub fn error(error: Error) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Error,
        error: Some(error.to_string()),
        ..Default::default()
    }
}
