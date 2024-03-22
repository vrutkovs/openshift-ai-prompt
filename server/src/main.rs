use futures_util::{stream::SplitSink, SinkExt, StreamExt};
use log::*;
use openshift_ai_prompt_common::{models, ws};
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{
    accept_async,
    tungstenite::{Error, Message, Result},
    WebSocketStream,
};

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
                        let msg = msg?;
                        if msg.is_close() {
                            continue
                        }
                        match msg.as_msg() {
                            Ok(m) => handle_msg(m, peer, &mut ws_sender).await?,
                            Err(e) => ws_sender.send(ws::error(e).as_msg()).await?,
                        }
                    }
                    None => break,
                }
            }
        }
    }

    Ok(())
}

async fn handle_msg(
    m: ws::WSMessage,
    peer: SocketAddr,
    ws_sender: &mut SplitSink<WebSocketStream<TcpStream>, Message>,
) -> Result<(), Error> {
    match m.msgtype {
        ws::WSMessageType::Prompt => match k8s_job::start(
            peer.to_string(),
            m.message.unwrap_or(String::from("")),
            m.model.unwrap(),
            ws_sender,
        )
        .await
        {
            Err(e) => ws_sender.send(ws::error(e).as_msg()).await,
            Ok(url) => ws_sender.send(ws::result(url).as_msg()).await,
        },
        _ => {
            ws_sender
                .send(ws::error(anyhow::format_err!("unexpected error type")).as_msg())
                .await
        }
    }
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
