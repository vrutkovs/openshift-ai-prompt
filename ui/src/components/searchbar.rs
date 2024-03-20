use patternfly_yew::prelude::*;
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
                let prompt_input_element = self.input_ref.cast::<HtmlInputElement>().unwrap();
                let prompt = prompt_input_element.value();
                ctx.props().on_generate.emit(prompt.clone().into());
                self.input = prompt.clone();
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <Split>
                <SplitItem  fill=true>
                    <TextInputGroup>
                        <TextInputGroupMain
                            placeholder="Placeholder"
                            icon={Icon::Search}
                            value={ self.input.clone() }
                        />
                        <TextInputGroupUtilities>
                            <Button icon={Icon::Times} variant={ButtonVariant::Plain} onclick={ctx.link().callback(|_| Msg::Generate)} />
                        </TextInputGroupUtilities>
                    </TextInputGroup>

                // <input  type="text" class="bg-white flex-grow h-12 px-4 rounded-lg focus:outline-none hover:cursor-pointer" name=""/>
                // <button class="btn btn-blue w-1/10 h-12 px-4 flew-grow-0" onclick={ctx.link().callback(|_| Msg::Generate)}>
                // <i class="fa-solid fa-search"></i>
                // </button>
                </SplitItem>
            </Split>
        }
    }
}
