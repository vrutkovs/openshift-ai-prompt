use openshift_ai_prompt_common::models::Model;
use openshift_ai_prompt_common::ws;
use openshift_ai_prompt_common::ws::AsWSMessage;

use yew::platform::spawn_local;
use yew::prelude::*;

use futures::{SinkExt, StreamExt};
use gloo::console;
use reqwasm::websocket::{events::CloseEvent, futures::WebSocket, WebSocketError};

pub fn generate_image(
    prompt: String,
    model: Model,
    progress: Callback<(AttrValue, f64)>,
    error: Callback<AttrValue>,
    result: Callback<AttrValue>,
) {
    progress.emit((AttrValue::from("Initializing"), 0.1));

    let window = web_sys::window().expect("Missing Window");
    let host = window.location().host().expect("Missing href");
    let ws_href = format!("wss://{}/ws", host);
    console::log!(
        "detected host '%s', ws address: '%s'",
        host,
        ws_href.clone()
    );

    let ws_address = match std::option_env!("PRODUCTION") {
        Some("true") => ws_href.as_str(),
        Some(&_) | None => "ws://127.0.0.1:8081",
    };
    let ws = WebSocket::open(ws_address).expect("WebSocket open");
    let (mut write, mut read) = ws.split();

    spawn_local(async move {
        match ws::prompt(prompt, model) {
            Err(e) => error.emit(AttrValue::from(format!("{:?}", e))),
            Ok(msg) => write.send(msg).await.unwrap(),
        };
        while let Some(msg) = read.next().await {
            match msg.as_msg() {
                Ok(ws_message) => match ws_message.msgtype {
                    ws::WSMessageType::Progress => {
                        let text = AttrValue::from(ws_message.to_string());
                        let percent = ws_message.get_progress().unwrap_or(0.3);
                        progress.emit((text, percent))
                    }
                    ws::WSMessageType::Result => {
                        result.emit(AttrValue::from(ws_message.to_string()))
                    }
                    ws::WSMessageType::Error => error.emit(AttrValue::from(ws_message.to_string())),
                    ws::WSMessageType::Prompt => {
                        error.emit(AttrValue::from("prompt received".to_string()))
                    }
                },
                Err(e) => {
                    if let Some(WebSocketError::ConnectionClose(_)) = msg.err() {
                        continue;
                    }
                    error.emit(AttrValue::from(format!("{:?}", e)));
                }
            }
        }
    })
}
