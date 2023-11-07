import React, { useEffect } from 'react';
import './App.css';

// Import the wasm module
import init, { greet, Person } from './pkg/test_wasm2';

 

function App() {
  useEffect(() => {
    // Initialize the wasm module
    init()
      .then(() => {
        // Use the wasm functions after the module is initialized
        greet('React');

        const person = new Person("Alice", 30);
        console.log(person.name); // Should output "Alice"
        person.celebrate_birthday();
        console.log(`${person.name} is now ${person.age} years old`);
      })
      .catch(e => console.error("Error importing `wasm_example.js`:", e));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>OK</h1>
        <p>
          
          Edit <code>src/App.js</code> and save to reload.
        </p>
      </header>
    </div>
  );
}

export default App;
