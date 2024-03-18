use openshift_ai_prompt_common::ws;
use openshift_ai_prompt_common::ws::AsWSMessage;

use yew::platform::spawn_local;
use yew::prelude::*;

use futures::{SinkExt, StreamExt};
use reqwasm::websocket::{futures::WebSocket, Message as reqwasm_Message};

pub fn generate_image(
    progress: Callback<(AttrValue, f32)>,
    error: Callback<AttrValue>,
    result: Callback<AttrValue>,
) {
    progress.emit((AttrValue::from("Initializing"), 0.1));
    let mut ws_address = "ws://127.0.0.1:8081";
    if std::option_env!("PRODUCTION") == Some("true") {
        ws_address = "ws://127.0.0.1:9090/ws"
    }
    let ws = WebSocket::open(ws_address).unwrap();
    let (mut write, mut read) = ws.split();

    spawn_local(async move {
        write
            .send(reqwasm_Message::Text("start".to_string()))
            .await
            .unwrap();
        while let Some(msg) = read.next().await {
            match msg.as_msg() {
                Ok(ws_message) => match ws_message.msgtype {
                    ws::WSMessageType::Progress => {
                        progress.emit((AttrValue::from(ws_message.to_string()), 0.5))
                    }
                    ws::WSMessageType::Result => {
                        result.emit(AttrValue::from(ws_message.to_string()))
                    }
                    ws::WSMessageType::Error => error.emit(AttrValue::from(ws_message.to_string())),
                },
                Err(e) => {
                    progress.emit((AttrValue::from(format!("{:?}", e)), 0.5));
                }
            }
        }
    })
}
