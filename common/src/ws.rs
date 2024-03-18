use anyhow::Error;
use reqwasm::websocket::{Message as reqwasm_Message, WebSocketError};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Default, Deserialize, Serialize)]
pub enum WSMessageType {
    #[default]
    Progress,
    Result,
    Error,
}

#[derive(Default, Deserialize, Serialize)]
pub struct WSMessage {
    #[serde(rename(serialize = "type", deserialize = "type"))]
    pub msgtype: WSMessageType,
    pub message: Option<String>,
    pub percentage: Option<f32>,
}

impl fmt::Display for WSMessage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message.clone().unwrap_or("".to_string()))
    }
}

impl WSMessage {
    pub fn get_progress(&self) -> Option<f32> {
        self.percentage
    }
}

pub trait AsWSMessage {
    fn as_msg(&self) -> Result<WSMessage, Error>;
}

impl AsWSMessage for Result<reqwasm_Message, WebSocketError> {
    fn as_msg(&self) -> Result<WSMessage, Error> {
        match &self {
            Ok(msg) => match msg {
                reqwasm_Message::Bytes(_) => anyhow::bail!("bytes are not supported"),
                reqwasm_Message::Text(text) => Ok(serde_json::from_str(text)?),
            },
            Err(e) => anyhow::bail!(format!("{:?}", e)),
        }
    }
}

pub fn progress(status: &str, percentage: f32) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Progress,
        message: Some(status.to_string()),
        percentage: Some(percentage),
    }
}

pub fn result(url: String) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Result,
        message: Some(url),
        ..Default::default()
    }
}

pub fn error(error: Error) -> WSMessage {
    WSMessage {
        msgtype: WSMessageType::Error,
        message: Some(error.to_string()),
        ..Default::default()
    }
}
