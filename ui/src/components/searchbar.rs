use patternfly_yew::prelude::*;
use web_sys::HtmlInputElement;
use yew::prelude::*;
use yew::Properties;
pub enum Msg {
    Generate,
    Select(Model),
}

use openshift_ai_prompt_common::models::{Model, MODELS};

#[derive(Clone, Properties, PartialEq)]
pub struct Props {
    pub on_generate: Callback<(AttrValue, Model)>,
}

pub struct SearchBar {
    input: String,
    model: Model,
    input_ref: NodeRef,
}
impl Component for SearchBar {
    type Message = Msg;
    type Properties = Props;

    fn create(_: &Context<Self>) -> Self {
        Self {
            input: String::from("A person wearing red fedora in style of Picasso"),
            model: Model::StableDiffusionXL,
            input_ref: NodeRef::default(),
        }
    }

    fn update(&mut self, ctx: &yew::Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::Generate => {
                let prompt_input_element = self.input_ref.cast::<HtmlInputElement>().unwrap();
                let prompt = prompt_input_element.value();
                ctx.props()
                    .on_generate
                    .emit((prompt.clone().into(), self.model));
                self.input = prompt.clone();
                true
            }
            Msg::Select(m) => {
                self.model = m;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <Split>
                <SplitItem fill=true>
                    <TextInputGroup>
                        <TextInputGroupMain
                            placeholder="Placeholder"
                            value={ self.input.clone() }
                            inner_ref = {self.input_ref.clone()}
                        />
                        <TextInputGroupUtilities>
                            <Button icon={Icon::Search} variant={ButtonVariant::Plain} onclick={ctx.link().callback(|_| Msg::Generate)} />
                        </TextInputGroupUtilities>
                    </TextInputGroup>
                </SplitItem>
                <SplitItem>
                    <Dropdown
                        variant={MenuToggleVariant::Plain}
                        icon={Icon::EllipsisV}
                    >
                    {for MODELS.iter().map(|m| {
                        let onclick = ctx.link().callback(move |_| Msg::Select(*m));
                        let mut icon = None;
                        if *m == self.model {
                            icon = Some(html!(Icon::CheckCircle))
                        }
                        html_nested!(<MenuAction {icon} {onclick}>
                            <span>{m.name()}</span>
                        </MenuAction>)
                    })}
                    </Dropdown>
                </SplitItem>
            </Split>
        }
    }
}
