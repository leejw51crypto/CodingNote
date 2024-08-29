// SPDX-License-Identifier: MIT
// This uses v4.9.5 of the OpenZeppelin Contracts
// https://github.com/OpenZeppelin/openzeppelin-contracts/tree/v4.9.5

pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title MyERC20Token
 * @dev This is a basic ERC20 token using OpenZeppelin's templates.
 * You can edit the default values as needed.
 */
contract MyERC20Token is ERC20Burnable, ERC20Pausable, AccessControl {
    bytes32 public constant CONTROLLER_ROLE = keccak256("CONTROLLER_ROLE");

    /**
     * @dev Constructor to initialize the token with default values.
     * You can edit these values as needed.
     */
    constructor() ERC20("My token name", "MyTokenSymbol") {
        // Default initial supply of 1 million tokens (with 18 decimals)
        uint256 initialSupply = 1_000_000 * (10 ** 18);

        // The initial supply is minted to the deployer's address
        _mint(msg.sender, initialSupply);

        // The deployer is granted the default admin role and the controller role
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(CONTROLLER_ROLE, msg.sender);
    }

    // Additional functions or overrides can be added here if needed.
    function pause() public onlyRole(CONTROLLER_ROLE) {
        _pause();
    }

    function unpause() public onlyRole(CONTROLLER_ROLE) {
        _unpause();
    }

    function mint(address to, uint256 amount) public onlyRole(CONTROLLER_ROLE) {
        _mint(to, amount);
    }

    /**
     * @dev Internal functions
     */

    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal virtual override(ERC20, ERC20Pausable) {
        super._beforeTokenTransfer(from, to, amount);
    }
}
