use yew::prelude::*;

#[derive(Clone, Properties, PartialEq)]
pub struct Props {
    pub message: AttrValue,
    pub percent: f32,
    pub error: bool,
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
        let error = ctx.props().error;
        html! {
            <div class="w-full">
                <div class="bg-stroke bg-dark-3 relative h-4 w-full">
                    if !message.is_empty() {
                        if error {
                            <div class="bg-red-500 text-white font-bold rounded-t px-4 py-2 m-2" role="alert">
                                <pre class="text-sm">{message}</pre>
                            </div>
                        } else {
                            <div class="bg-blue-100 border-t border-b border-blue-500 text-blue-700 px-4 py-3" role="alert">
                                <p class="text-sm">{message}</p>
                            </div>
                        }
                    }
                    if percent > 0.0 && percent != 1.0 {
                        <progress class="bg-primary m-2 left-0 top-0 flex h-full w-full items-center justify-center rounded-2xl text-black" value={percent.to_string()} max=1.0></progress>
                    }
                </div>
            </div>
        }
    }
}
