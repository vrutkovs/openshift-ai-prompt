use gloo::console;
use yew::prelude::*;

use crate::progress::ProgressBar;
use crate::searchbar::SearchBar;

#[derive(Debug, Default)]
pub struct App;

pub enum Msg {
    Prompt(AttrValue),
    Progress,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_: &Context<Self>) -> Self {
        Self
    }

    fn update(&mut self, _: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Prompt(prompt) => {
                console::log!("generating prompt %s", prompt.as_str());
                true
            }
            Msg::Progress => {
                console::log!("progress");
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let on_generate = ctx.link().callback(Msg::Prompt);
        html! {
          <div class="flex flex-col h-screen">
            <nav class="h-screen items-center px-4 py-4 bg-gray-300 flex-col">
              <SearchBar {on_generate}/>
            </nav>
            <div>
              <ProgressBar />
            </div>
          </div>
        }
    }
}
