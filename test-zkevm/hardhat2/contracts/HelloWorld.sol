// SPDX-License-Identifier: MIT


pragma solidity ^0.8.0;
contract HelloWorld {
    string public greet = "Hello World!";

    function setGreeting(string memory _greeting) public {
        greet = _greeting;
    }

    function getGreeting() public view returns (string memory) {
        return greet;
    }
}