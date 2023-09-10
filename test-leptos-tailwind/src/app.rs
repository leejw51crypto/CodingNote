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
                <Route path="/login" view=  move || view! { <Login/> }/>
                <Route path="/signup" view=  move || view! { <Signup/> }/>
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
        <div class="bg-gray-100 min-h-screen p-5">

        <img src="https://cdn2.thecatapi.com/images/6eg.jpg" class="w-32 h-32 rounded-2xl" alt="cat"  />

        <div class="bg-white shadow-md rounded p-5 mb-4">
            <h1 class="text-2xl font-bold">About</h1>
        </div>
    
        <div class="bg-white shadow-md rounded p-5 mb-4">
            <h2 class="text-xl mb-4">Basic Info</h2>
            <ul class="space-y-2">
                <li><strong>Stellar Birth Cycle:</strong> Galactic Year 5000</li>
                <li><strong>Species:</strong> Zorbonian</li>
                <li><strong>Interested in:</strong> Star Gazing, Black Hole Diving, Quantum Computing</li>
            </ul>
        </div>
    
        <div class="bg-white shadow-md rounded p-5 mb-4">
            <h2 class="text-xl mb-4">Work and Education</h2>
            <ul class="space-y-2">
                <li><strong>Current Space Station:</strong> Andromeda Hub</li>
                <li><strong>Position:</strong> Quantum Code Manipulator</li>
                <li><strong>Interstellar College:</strong> Nebula Tech</li>
                <li><strong>High School:</strong> Zorbon Prep School</li>
            </ul>
        </div>
    
      
    </div>
    
      
    }
}

#[component]    
fn Login() -> impl IntoView {

    view! {
        <div class="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
        <div class="max-w-md w-full space-y-8 bg-white p-6 rounded-xl shadow-md">
          <div>
            <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
              Sign in to your account
            </h2>
          </div>
          <form class="mt-8 space-y-6" action="#" method="POST">
            <input type="hidden" name="remember" value="true"/>
            <div class="rounded-md shadow-sm -space-y-px">
              <div>
                <label for="email-address" class="sr-only">Email address</label>
                <input id="email-address" name="email" type="email" autocomplete="email" required class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm" placeholder="Email address"/>
              </div>
              <div>
                <label for="password" class="sr-only">Password</label>
                <input id="password" name="password" type="password" autocomplete="current-password" required class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm" placeholder="Password"/>
              </div>
            </div>
      
            <div class="flex items-center justify-between">
              <div class="flex items-center">
                <input id="remember-me" name="remember-me" type="checkbox" class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"/>
                <label for="remember-me" class="ml-2 block text-sm text-gray-900">
                  Remember me
                </label>
              </div>
      
              <div class="text-sm">
                <a href="#" class="font-medium text-indigo-600 hover:text-indigo-500">
                  Forgot your password?
                </a>
              </div>
            </div>
      
            <div>
              <button type="submit" class="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                Sign In
              </button>
            </div>
          </form>
        </div>
      </div>
      
    }
}


#[component]
fn Navbar() -> impl IntoView {
    
    view! {
      
        <nav class="bg-blue-500 p-4 flex justify-between">
            <div class="text-white font-bold text-xl">
                "LeptosApp"
            </div>
            <div class="space-x-4">
            <button class="transition-all duration-300 ease-in-out bg-violet-500 border-2 border-violet-700 text-white hover:bg-violet-600 hover:border-violet-800 font-medium py-2 px-4 rounded-full" on:click=move |_| use_navigate()("/", Default::default())>
            Home
        </button>
        
        <button class="transition-all duration-300 ease-in-out bg-violet-500 border-2 border-violet-700 text-white hover:bg-violet-600 hover:border-violet-800 font-medium py-2 px-4 rounded-full" on:click=move |_| use_navigate()("/about", Default::default())>
            About
        </button>
        
        
        <button class="transition-all duration-300 ease-in-out bg-violet-500 border-2 border-violet-700 text-white hover:bg-violet-600 hover:border-violet-800 font-medium py-2 px-4 rounded-full" on:click=move |_| use_navigate()("/login", Default::default())>
        Login
    </button>

    <button class="transition-all duration-300 ease-in-out bg-violet-500 border-2 border-violet-700 text-white hover:bg-violet-600 hover:border-violet-800 font-medium py-2 px-4 rounded-full" on:click=move |_| use_navigate()("/signup", Default::default())>
    Signup
</button>
            </div>
        </nav>
    }
}





#[component]
fn Signup() -> impl IntoView {

    view! {
        <div class="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
        <div class="max-w-md w-full space-y-8 bg-white p-6 rounded-xl shadow-md">
          <div>
            <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
              Create a new account
            </h2>
          </div>
          <form class="mt-8 space-y-6" action="#" method="POST">
            <input type="hidden" name="remember" value="true"/>
            <div class="rounded-md shadow-sm -space-y-px">
              <div>
                <label for="email-address-signup" class="sr-only">Email address</label>
                <input id="email-address-signup" name="email" type="email" autocomplete="email" required class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm" placeholder="Email address"/>
              </div>
              <div>
                <label for="password-signup" class="sr-only">Password</label>
                <input id="password-signup" name="password" type="password" autocomplete="new-password" required class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm" placeholder="Password"/>
              </div>
              <div>
                <label for="confirm-password-signup" class="sr-only">Confirm Password</label>
                <input id="confirm-password-signup" name="confirm-password" type="password" autocomplete="new-password" required class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm" placeholder="Confirm Password"/>
              </div>
            </div>

            <div>
              <button type="submit" class="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                Sign Up
              </button>
            </div>
          </form>
        </div>
      </div>
    }
}
