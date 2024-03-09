use web_sys::HtmlInputElement;
use yew::prelude::*;
use yew::Properties;

pub enum Msg {
    Generate,
}

#[derive(Clone, Properties, PartialEq)]
pub struct Props {
    pub on_generate: Callback<AttrValue>,
}

pub struct SearchBar {
    input: String,
    input_ref: NodeRef,
}
impl Component for SearchBar {
    type Message = Msg;
    type Properties = Props;

    fn create(_: &Context<Self>) -> Self {
        Self {
            input: String::from("A person wearing red fedora in style of Picasso"),
            input_ref: NodeRef::default(),
        }
    }

    fn update(&mut self, ctx: &yew::Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Generate => {
                let value_input_element = self.input_ref.cast::<HtmlInputElement>().unwrap();
                let new_value = value_input_element.value();
                ctx.props().on_generate.emit(new_value.clone().into());
                self.input = new_value.clone();
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <nav class="items-center px-4 py-4 bg-gray-300 flex">
                <input ref={self.input_ref.clone()} type="text" class="bg-white flex-grow h-12 px-4 rounded-lg focus:outline-none hover:cursor-pointer" name="" value={ self.input.clone() }/>
                <button class="btn btn-blue w-1/10 h-12 px-4 flew-grow-0" onclick={ctx.link().callback(|_| Msg::Generate)}>
                <i class="fa-solid fa-search"></i>
                </button>
            </nav>
        }
    }
}
