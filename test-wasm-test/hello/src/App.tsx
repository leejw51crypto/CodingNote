import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';
// Import the WebAssembly module
import * as wasm from './pkg/mywasm';

function App() {
  const [isWasmLoaded, setIsWasmLoaded] = useState(false);

  // use effect is initialize function
  useEffect(() => {
    // wasm.default loads wasm asynchronously
    wasm.default().then(() => {
      setIsWasmLoaded(true);
    });
  }, []); // without argument, one time initialize

  // Event handler for the button
  const handleButtonClick = () => {
    if (isWasmLoaded) {
      wasm.greet();
      let a= wasm.myadd(1,2);
      alert(a);
    } else {
      console.log("WebAssembly module not loaded yet.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        {/* Add a button here */}
        <button onClick={handleButtonClick} disabled={!isWasmLoaded}>
          Greet
        </button>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
