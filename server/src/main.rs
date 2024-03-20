use futures_util::SinkExt;
use futures_util::StreamExt;
use log::*;
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{
    accept_async,
    tungstenite::{Error, Message, Result},
};

use openshift_ai_prompt_common::models;
use openshift_ai_prompt_common::ws;
mod k8s_job;
use crate::k8s_job::{AsTungstenite, AsWSMessage};

async fn accept_connection(peer: SocketAddr, stream: TcpStream) {
    if let Err(e) = handle_connection(peer, stream).await {
        match e {
            Error::ConnectionClosed | Error::Protocol(_) | Error::Utf8 => (),
            err => error!("Error processing connection: {}", err),
        }
    }
}

async fn handle_connection(peer: SocketAddr, stream: TcpStream) -> Result<()> {
    let ws_stream = accept_async(stream).await.expect("Failed to accept");
    info!("New WebSocket connection: {}", peer);
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Wait for message to be received
    loop {
        tokio::select! {
            msg = ws_receiver.next() => {
                match msg {
                    Some(msg) => {
                        match msg?.as_msg() {
                            Ok(m) => match m.msgtype {
                                ws::WSMessageType::Prompt => match k8s_job::start(peer.to_string(), m.message.unwrap_or(String::from("")), m.model.unwrap(), &mut ws_sender).await {
                                    Err(e) => ws_sender.send(ws::error(e).as_msg()).await?,
                                    Ok(url) => ws_sender.send(ws::result(url).as_msg()).await?,
                                },
                                _ => todo!(),
                            },
                            Err(_) => todo!(),
                        }
                    }
                    None => break,
                }
            }
        }
    }

    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() {
    env_logger::init();

    let addr = "0.0.0.0:8081";
    let listener = TcpListener::bind(&addr).await.expect("Can't listen");
    info!("Listening on: {}", addr);

    while let Ok((stream, _)) = listener.accept().await {
        let peer = stream
            .peer_addr()
            .expect("connected streams should have a peer address");

        tokio::spawn(accept_connection(peer, stream));
    }
}
