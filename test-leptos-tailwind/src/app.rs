use leptos::*;
use leptos_meta::*;
use leptos_router::*;

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {

        <Stylesheet id="leptos" href="/pkg/tailwind.css"/>
        <Link rel="shortcut icon" type_="image/ico" href="/favicon.ico"/>
        <Router>
            <Navbar/> 
            <Routes>
                <Route path="" view=  move || view! { <Home/> }/>
                <Route path="/about" view=  move || view! { <About/> }/>
            </Routes>
        </Router>
    }
}

#[component]
fn Home() -> impl IntoView {
    let (count, set_count) = create_signal(0);

    view! {
        <main class="my-0 mx-auto max-w-3xl text-center">
            <h2 class="p-6 text-4xl">"Welcome to Leptos with Tailwind"</h2>
            <p class="px-10 pb-10 text-left">"Tailwind will scan your Rust files for Tailwind class names and compile them into a CSS file."</p>
            <button
                class="bg-amber-600 hover:bg-sky-700 px-5 py-3 text-white rounded-lg"
                on:click=move |_| set_count.update(|count| *count += 1)
            >
                "Something's here | "
                {move || if count() == 0 {
                    "Click me!".to_string()
                } else {
                    count().to_string()
                }}
                " | Some more text"
            </button>
        </main>
    }
}


#[component]    
fn About() -> impl IntoView {

    view! {
        <main class="my-0 mx-auto max-w-3xl text-center">
            <h2 class="p-6 text-4xl">"Welcome to Leptos with Tailwind"</h2>
            <p class="px-10 pb-10 text-left">"about"</p>
           
        </main>
    }
}


#[component]
fn Navbar() -> impl IntoView {
    let navigate = use_navigate();
    let navigate2 = use_navigate();
    view! {
        <nav class="bg-blue-500 p-4 flex justify-between">
            <div class="text-white font-bold text-xl">
                "LeptosApp"
            </div>
            <div class="space-x-4">
            <button class="transition-all duration-300 ease-in-out bg-violet-500 border-2 border-violet-700 text-white hover:bg-violet-600 hover:border-violet-800 font-medium py-2 px-4 rounded-full" on:click=move |_| navigate2("/", Default::default())>
            Home
        </button>
        
        <button class="transition-all duration-300 ease-in-out bg-violet-500 border-2 border-violet-700 text-white hover:bg-violet-600 hover:border-violet-800 font-medium py-2 px-4 rounded-full" on:click=move |_| navigate("/about", Default::default())>
            About
        </button>
        
        

            </div>
        </nav>
    }
}
