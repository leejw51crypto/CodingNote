pragma solidity ^0.8.2;

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";

contract MyErc1155 is ERC1155 {
    uint256 public constant GOLD = 0;
    uint256 public constant SILVER = 1;
    uint256 public constant IRON = 2;
    uint256 public constant DIAMOND = 3;
    uint256 public constant RUBY = 4;

    constructor() public ERC1155("https://game.example/api/item/{id}.json") {
        _mint(msg.sender, GOLD, 10001, "gold");
        _mint(msg.sender, SILVER, 20002, "silver");
        _mint(msg.sender, IRON, 30003, "iron");
        _mint(msg.sender, DIAMOND, 40004, "diamond");
        _mint(msg.sender, RUBY, 50005, "ruby");
    }
}