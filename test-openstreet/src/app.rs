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
            <Routes>
                <Route path="" view=  move || view! { <Home/> }/>
            </Routes>
        </Router>
    }
}

#[component]
fn Home() -> impl IntoView {
    let (count, set_count) = create_signal(0);
    let (map_location, set_map_location) = create_signal((36.1699, -115.1879)); // Las Vegas coordinates

    view! {
        <div class="my-0 mx-auto max-w-3xl text-center bg-white rounded-2xl shadow-lg p-8">
            <h2 class="p-6 text-4xl font-bold text-gray-800">"Welcome to Leptos with Tailwind"</h2>

            <div class="mt-4">
                <button
                    class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-xl mr-4"
                    on:click=move |_| set_map_location((36.12129, -115.16217)) // Las Vegas coordinates
                >
                    "Las Vegas"
                </button>
                <button
                    class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-xl"
                    on:click=move |_| set_map_location((-33.85697, 151.21505)) // Sydney coordinates
                >
                    "Sydney"
                </button>
            </div>

            <div class="mt-8">
                <div class="bg-gradient-to-r from-blue-400 to-blue-600 p-4 rounded-2xl shadow-2xl inline-block transform transition duration-500 hover:scale-105">
                    <p class="text-lg font-semibold text-white">"Latitude: " {move || map_location.get().0}</p>
                    <p class="text-lg font-semibold text-white">"Longitude: " {move || map_location.get().1}</p>
                </div>
            </div>

            <div class="mt-8">
                <OpenStreetMap
                    map_location=map_location
                />
            </div>
        </div>
    }
}

#[component]
pub fn OpenStreetMap(map_location: ReadSignal<(f64, f64)>) -> impl IntoView {
    let generate_map_url = move || {
        let (lat, lon) = map_location.get();
        let bbox = format!(
            "{},{},{},{}",
            lon - 0.002,
            lat - 0.002,
            lon + 0.002,
            lat + 0.002
        );
        let marker = format!("{lat},{lon}", lat = lat, lon = lon);

        format!("https://www.openstreetmap.org/export/embed.html?bbox={bbox}&layer=mapnik&marker={marker}")
    };

    view! {
        <iframe
            src=generate_map_url
            style="border: 1px solid black"
            width="100%"
            height="500"
        ></iframe>
    }
}
