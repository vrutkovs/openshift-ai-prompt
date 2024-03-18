use openshift_ai_prompt_common::ws;
use openshift_ai_prompt_common::ws::AsWSMessage;

use yew::platform::spawn_local;
use yew::prelude::*;

use futures::{SinkExt, StreamExt};
use reqwasm::websocket::{futures::WebSocket, Message as reqwasm_Message};

pub fn generate_image(cb: Callback<(AttrValue, f32)>, result: Callback<AttrValue>) {
    cb.emit((AttrValue::from("Initializing"), 0.1));
    let ws = WebSocket::open("ws://127.0.0.1:8081").unwrap();
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
                        cb.emit((AttrValue::from(ws_message.to_string()), 0.5))
                    }
                    ws::WSMessageType::Result => {
                        result.emit(AttrValue::from(ws_message.to_string()))
                    }
                    ws::WSMessageType::Error => {
                        cb.emit((AttrValue::from(ws_message.to_string()), 0.5))
                    }
                },
                Err(e) => {
                    cb.emit((AttrValue::from(format!("{:?}", e)), 0.5));
                }
            }
        }
    })
}
