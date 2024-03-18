use gloo::console;
use yew::prelude::*;

use crate::components::picture::Picture;
use crate::components::progress::ProgressBar;
use crate::components::searchbar::SearchBar;
use crate::generate;

#[derive(Debug, Default)]
pub struct App {
    status: String,
    progress: f32,
    error: bool,
    result: Option<String>,
}

pub enum Msg {
    Prompt(AttrValue),
    Progress((AttrValue, f32)),
    Error(AttrValue),
    Result(AttrValue),
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_: &Context<Self>) -> Self {
        Self {
            status: String::from(""),
            progress: 0.0,
            result: None,
            error: false,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Prompt(prompt) => {
                console::log!("generating prompt %s", prompt.as_str());

                self.result = None;
                let on_progress = ctx.link().callback(Msg::Progress);
                let on_error = ctx.link().callback(Msg::Error);
                let on_result = ctx.link().callback(Msg::Result);
                generate::generate_image(on_progress, on_error, on_result);
                true
            }
            Msg::Progress((status, progress)) => {
                self.status = status.to_string();
                self.progress = progress;
                true
            }
            Msg::Error(error) => {
                self.status = error.to_string();
                self.error = true;
                true
            }
            Msg::Result(url) => {
                self.result = Some(url.to_string());
                self.progress = 0.0;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let on_generate = ctx.link().callback(Msg::Prompt);

        html! {
          <div class="flex flex-col h-screen">
            <SearchBar {on_generate}/>
            <ProgressBar message={self.status.clone()} percent={self.progress} error={self.error} />
            <Picture url={self.result.clone()}/>
          </div>
        }
    }
}
