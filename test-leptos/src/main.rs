use leptos::*;

fn main2() {
    mount_to_body(|cx| view! { cx,  <p>"Hello, world!"</p> })
}

fn main() {
    _ = console_log::init_with_level(log::Level::Debug);
    console_error_panic_hook::set_once();
    leptos::mount_to_body(|cx| view! { cx, <App/> })
}

#[component]
fn App(cx: Scope) -> impl IntoView {
    
    let (count, set_count) = create_signal(cx, 0);
    create_effect(cx, move |_| {
        log::info!("Debug Value: {}", count());
    });
    view! { cx,
        <button
            on:click=move |_| {
                set_count(3);
            }
        >
            "Click me: "
            {move || count.get()}
        </button>
    }
}