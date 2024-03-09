use yew::prelude::*;
use gloo::console;

#[derive(Debug, Default)]
pub struct App {
    prompt: String,
}

pub enum Msg {
    Prompt,
    Progress
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_: &Context<Self>) -> Self {
        Self {
          prompt: "A person wearing red fedora in style of Picasso".to_string()
        }
    }

    fn update(&mut self, _: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Prompt => {
                console::log!("generating prompt %s", self.prompt.clone());
                true
            }
            Msg::Progress => {
                console::log!("progress");
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
          <div class="flex flex-col h-screen">
            <nav class="h-screen items-center px-4 py-4 bg-gray-300 flex-col">
              <form class="w-full flex mb-4">
                <input type="text" class="bg-white w-full h-12 px-4 rounded-lg focus:outline-none hover:cursor-pointer" name="" value={ self.prompt.clone() }/>
                <button class="btn btn-blue w-1/10 h-12 px-4" onclick={ctx.link().callback(|_| Msg::Prompt)}>
                  <i class="fa-solid fa-search"></i>
                </button>
              </form>
            </nav>
          </div>
        }
    }
}
