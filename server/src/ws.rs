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

pub fn status_message(status: &str) -> Message {
    let msg = WSMessage {
        msgtype: WSMessageType::Progress,
        status: Some(status.to_string()),
        ..Default::default()
    };
    match serde_json::to_string(&msg) {
        Ok(j) => Message::Text(j.to_owned()),
        Err(e) => Message::Text(e.to_string()),
    }
}

pub fn result(url: String) -> Message {
    let msg = WSMessage {
        msgtype: WSMessageType::Result,
        url: Some(url),
        ..Default::default()
    };
    match serde_json::to_string(&msg) {
        Ok(j) => Message::Text(j.to_owned()),
        Err(e) => Message::Text(e.to_string()),
    }
}

pub fn error_message(error: Error) -> Message {
    let msg = WSMessage {
        msgtype: WSMessageType::Error,
        error: Some(error.to_string()),
        ..Default::default()
    };
    match serde_json::to_string(&msg) {
        Ok(j) => Message::Text(j.to_owned()),
        Err(e) => Message::Text(e.to_string()),
    }
}
