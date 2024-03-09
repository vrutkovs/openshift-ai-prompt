#![recursion_limit = "1024"]

use console_error_panic_hook::set_once as set_panic_hook;

mod app;
mod progress;
mod searchbar;

use app::App;

fn main() {
    set_panic_hook();

    yew::Renderer::<App>::new().render();
}
