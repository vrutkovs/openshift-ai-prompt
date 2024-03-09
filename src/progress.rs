use yew::prelude::*;

#[derive(Clone, Properties, PartialEq)]
pub struct Props {
    pub message: AttrValue,
    pub percent: f32,
}

pub struct ProgressBar;
impl Component for ProgressBar {
    type Message = ();
    type Properties = Props;

    fn create(_: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let message = &ctx.props().message;
        let percent = ctx.props().percent;
        html! {
            <div class="m-4 w-full">
                if percent > 0.0 && percent != 1.0 {
                    <div class="bg-stroke bg-dark-3 relative h-4 w-full rounded-2xl">
                        <progress class="bg-primary absolute left-0 top-0 flex h-full w-full items-center justify-center rounded-2xl text-black" value={percent.to_string()} max=1.0></progress>
                        <div class="bg-primary absolute mt-4 items-center">{message}</div>
                    </div>
                }
            </div>
        }
    }
}
