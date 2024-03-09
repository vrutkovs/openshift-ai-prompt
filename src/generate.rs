use std::time::Duration;
use yew::platform::spawn_local;
use yew::platform::time::sleep;
use yew::prelude::*;

const ONE_SEC: Duration = Duration::from_secs(1);

pub fn generate_image(cb: Callback<(AttrValue, f32)>, result: Callback<AttrValue>) {
    spawn_local(async move {
        cb.emit((AttrValue::from("Initializing"), 0.1));
        sleep(ONE_SEC).await;
        cb.emit((AttrValue::from("Still going"), 0.5));
        sleep(ONE_SEC).await;
        cb.emit((AttrValue::from("Done"), 1.0));

        result.emit(AttrValue::from("https://vrutkovs.eu/images/avatar.jpg"))
    })
}
