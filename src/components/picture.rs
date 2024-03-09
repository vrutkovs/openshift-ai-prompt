use yew::prelude::*;

#[derive(Clone, Properties, PartialEq)]
pub struct Props {
    pub url: Option<AttrValue>,
}

pub struct Picture;
impl Component for Picture {
    type Message = ();
    type Properties = Props;

    fn create(_: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let maybe_url = &ctx.props().url;
        if let Some(url) = maybe_url.as_ref() {
            html! {
                <div class="m-4 w-full">
                    <img class="h-512 w-512 object-scale-down" src={ url } />
                </div>
            }
        } else {
            html! {}
        }
    }
}
