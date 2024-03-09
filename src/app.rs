use yew::prelude::*;

#[derive(Debug, Default)]
pub struct App {
    prompt: String,
}

impl App {
}

impl Component for App {
    type Message = ();
    type Properties = ();

    fn create(_: Self::Properties, _: ComponentLink<Self>) -> Self {
        Self::default()
    }

    fn update(&mut self, _: Self::Message) -> bool {
        false
    }

    fn change(&mut self, _: Self::Properties) -> bool {
        false
    }

    fn view(&self) -> Html {
        html! {
          <div class="flex flex-col h-screen">
            <nav class="h-screen items-center px-4 py-4 bg-gray-300 flex-col">
              <form class="w-full flex mb-4">
                <input type="text" class="bg-white w-full h-12 px-4 rounded-lg focus:outline-none hover:cursor-pointer" name=""/>
                <button class="btn btn-blue w-1/10 h-12 px-4">
                  <i class="fa-solid fa-search"></i>
                </button>
              </form>
            </nav>
          </div>
        }
    }
}
