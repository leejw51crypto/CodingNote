-- ECDSA Wallet Manager in Lua
-- This script provides a menu-driven interface for wallet operations

-- Store the current wallet and mnemonic handle (in global scope for AI access)
current_wallet = nil
current_mnemonic_handle = nil

-- AI autorun setting (set to false to prompt before each execution)
ai_autorun = true

-- Function to print a separator line
local function print_separator()
	print("=" .. string.rep("=", 58))
end

-- Function to print the menu
local function print_menu()
	print_separator()
	print("            ECDSA WALLET MANAGER")
	print_separator()
	print("1. Generate New Mnemonic")
	print("2. Create Wallet from Mnemonic")
	print("3. Show Wallet Information")
	print("4. Show Private Key")
	print("5. Show Public Key")
	print("6. Show Addresses (Multiple)")
	print("7. Show Current Address")
	print("8. AI - Generate and Run Lua Code")
	print_separator()
	print("            CRONOS EVM FUNCTIONS")
	print_separator()
	print("9. Get Balance (Native Token)")
	print("10. Send Transaction (Native Token)")
	print("11. Get Transaction by Hash")
	print("12. Get Latest Block Number")
	print("13. Get ERC20 Token Balance")
	print("14. Send ERC20 Token")
	print("15. View/Update Config (RPC, Chain ID)")
	print("16. Set Default Address")
	print("17. Toggle AI Autorun (" .. (ai_autorun and "ON" or "OFF") .. ")")
	print("0. Exit")
	print_separator()
	io.write("Enter your choice: ")
	io.flush()
end

-- Function to generate a new mnemonic
local function generate_new_mnemonic()
	print("\n>> Generating new mnemonic...")
	current_mnemonic_handle = wallet.generate_mnemonic()
	print("\nNew Mnemonic Phrase:")
	print("-------------------")
	print(current_mnemonic_handle:get_mnemonic())
	print("\n⚠️  IMPORTANT: Save this mnemonic phrase securely!")
	print("⚠️  You will need it to recover your wallet.")
	print()

	io.write("Do you want to create a wallet with this mnemonic? (y/n): ")
	io.flush()
	local answer = io.read()

	if answer and (answer:lower() == "y" or answer:lower() == "yes") then
		io.write("Enter wallet index (default 0): ")
		io.flush()
		local index_str = io.read()
		local index = tonumber(index_str) or 0

		current_wallet = wallet.create_wallet(current_mnemonic_handle, index)
		print("\n✓ Wallet created successfully at index " .. index .. "!")
	end
end

-- Function to create wallet from existing mnemonic
local function create_wallet_from_mnemonic()
	print("\n>> Create Wallet from Mnemonic")
	local mnemonic_str = wallet.read_hidden("Please enter your mnemonic phrase (input hidden):\n> ")

	if not mnemonic_str or mnemonic_str == "" then
		print("❌ Error: Mnemonic cannot be empty")
		return
	end

	io.write("Enter wallet index (default 0): ")
	io.flush()
	local index_str = io.read()
	local index = tonumber(index_str) or 0

	local success, result = pcall(function()
		-- Import mnemonic to create handle
		local handle = wallet.import_mnemonic(mnemonic_str)
		return handle
	end)

	if success then
		current_mnemonic_handle = result
		local success2, wallet_result = pcall(function()
			return wallet.create_wallet(current_mnemonic_handle, index)
		end)

		if success2 then
			current_wallet = wallet_result
			print("✓ Wallet created successfully at index " .. index .. "!")
		else
			print("❌ Error creating wallet: " .. tostring(wallet_result))
		end
	else
		print("❌ Error: " .. tostring(result))
	end
end

-- Function to show full wallet information
local function show_wallet_info()
	if not current_wallet then
		print("\n❌ No wallet loaded. Please create or import a wallet first.")
		return
	end

	print("\n>> Wallet Information")
	print_separator()
	print("Wallet Index: " .. (current_wallet.wallet_index or 0))
	print()
	if current_mnemonic_handle then
		print("Mnemonic:")
		print(current_mnemonic_handle:get_mnemonic())
		print()
	end
	print("Private Key:")
	print(current_wallet.private_key)
	print()
	print("Public Key:")
	print(current_wallet.public_key)
	print()
	print("Address (ETH-style):")
	print(current_wallet.address)
	print_separator()
end

-- Function to show only private key
local function show_private_key()
	if not current_wallet then
		print("\n❌ No wallet loaded.")
		return
	end

	print("\n>> Private Key")
	print("-------------------")
	print(current_wallet.private_key)
	print()
end

-- Function to show only public key
local function show_public_key()
	if not current_wallet then
		print("\n❌ No wallet loaded.")
		return
	end

	print("\n>> Public Key")
	print("-------------------")
	print(current_wallet.public_key)
	print()
end

-- Function to show multiple addresses
local function show_multiple_addresses()
	if not current_wallet or not current_mnemonic_handle then
		print("\n❌ No wallet loaded.")
		return
	end

	io.write("How many addresses to display? (default 5): ")
	io.flush()
	local count_str = io.read()
	local count = tonumber(count_str) or 5

	print("\n>> Generating " .. count .. " addresses...")
	print_separator()

	local success, addresses = pcall(function()
		return wallet.generate_addresses(current_mnemonic_handle, count)
	end)

	if success then
		print("Derivation Path: m/44'/60'/0'/0/{index}")
		print()
		for i = 1, count do
			local addr = addresses[i]
			if addr then
				print(string.format("[%d] %s", addr.index, addr.address))
			end
		end
		print_separator()
	else
		print("❌ Error: " .. tostring(addresses))
	end
end

-- Function to show only current wallet address
local function show_address()
	if not current_wallet then
		print("\n❌ No wallet loaded.")
		return
	end

	print("\n>> Current Wallet Address")
	print("-------------------")
	print("Wallet Index: " .. (current_wallet.wallet_index or 0))
	print("Address: " .. current_wallet.address)
	print()
end

-- Function to handle AI code generation and execution
local function ai_mode()
	print("\n>> AI Code Generation Mode")
	print("-------------------")
	print("Enter your requests and the AI will generate and execute Lua code.")
	print("Type 'exit' to return to the main menu.")
	print("Type 'toggle' to toggle autorun mode.")
	print()
	print("Settings:")
	print("      - Autorun: " .. (ai_autorun and "ENABLED (auto-execute)" or "DISABLED (prompt before execute)"))
	print()
	print("Note: Variables from executed code persist across iterations.")
	print("      Global variables available:")
	if current_wallet then
		print("      - current_wallet (loaded: " .. current_wallet.address .. ")")
	else
		print("      - current_wallet (not loaded)")
	end
	if current_mnemonic_handle then
		print("      - current_mnemonic_handle (loaded)")
	else
		print("      - current_mnemonic_handle (not loaded)")
	end
	print()

	while true do
		io.write("AI> ")
		io.flush()
		local request = io.read()

		if not request or request:lower() == "exit" then
			print("\nExiting AI mode...")
			break
		end

		-- Handle toggle command
		if request:lower() == "toggle" then
			ai_autorun = not ai_autorun
			print(
				"\n>> Autorun mode: " .. (ai_autorun and "ENABLED (auto-execute)" or "DISABLED (prompt before execute)")
			)
			print()
			goto continue
		end

		if request:match("^%s*$") then
			-- Empty input, skip
			goto continue
		end

		-- Only show "Generating" message when autorun is disabled
		if not ai_autorun then
			print("\n>> Generating Lua code...")
		end

		local success, code = pcall(function()
			return wallet.generate_lua_code(request)
		end)

		if not success then
			print("Error generating code: " .. tostring(code))
			goto continue
		end

		-- Check autorun setting
		local should_execute = false
		if ai_autorun then
			should_execute = true
			-- In autorun mode, skip showing generated code and just execute
		else
			-- Show generated code when autorun is disabled
			print("\n>> Generated Code:")
			print("-------------------")
			print(code)
			print("-------------------")
			print()

			io.write("Execute this code? (y/n): ")
			io.flush()
			local confirm = io.read()
			should_execute = confirm and (confirm:lower() == "y" or confirm:lower() == "yes")
		end

		if should_execute then
			if not ai_autorun then
				print("\n>> Executing code...")
				print("-------------------")
			end
			print() -- Add blank line before output
			io.flush()

			local exec_success, exec_result = pcall(function()
				local chunk, load_err = load(code, "AI-generated", "t", _G)
				if not chunk then
					error("Failed to load code: " .. tostring(load_err))
				end
				return chunk()
			end)

			-- Only show execution status messages when autorun is disabled
			if not ai_autorun then
				print() -- Add blank line after output
				io.flush()

				if exec_success then
					print("-------------------")
					print(">> Code executed successfully!")
					if exec_result and exec_result ~= true then
						print(">> Return value: " .. tostring(exec_result))
					end
				else
					print("-------------------")
					print(">> Execution error: " .. tostring(exec_result))
				end
			else
				-- In autorun mode, only show errors
				if not exec_success then
					print("\n❌ Error: " .. tostring(exec_result))
				end
			end
		else
			if not ai_autorun then
				print("Skipping execution.")
			end
		end

		print()

		::continue::
	end
end

-- Function to get balance (native token)
local function cronos_get_balance_menu()
	print("\n>> Get Balance (Native Token)")
	print("-------------------")

	local config = wallet.get_config()
	local address = config.default_address

	if not address then
		io.write("Enter address to check (or press Enter to use current wallet): ")
		io.flush()
		local input = io.read()
		if input and input ~= "" then
			address = input
		elseif current_wallet then
			address = current_wallet.address
		else
			print("❌ No address provided")
			return
		end
	else
		io.write("Using default address: " .. address .. " (press Enter to continue or enter new address): ")
		io.flush()
		local input = io.read()
		if input and input ~= "" then
			address = input
		end
	end

	print("\n>> Fetching balance...")
	local success, balance = pcall(function()
		return wallet.cronos_get_balance(address)
	end)

	if success then
		print("\nAddress: " .. address)
		print("Balance: " .. balance .. " wei")
		print("Balance: " .. tonumber(balance) / 1e18 .. " tokens")
	else
		print("❌ Error: " .. tostring(balance))
	end
end

-- Function to send transaction (native token)
local function cronos_send_tx_menu()
	if not current_wallet then
		print("\n❌ No wallet loaded. Please create or import a wallet first.")
		return
	end

	print("\n>> Send Transaction (Native Token)")
	print("-------------------")

	io.write("Enter recipient address: ")
	io.flush()
	local to_address = io.read()

	if not to_address or to_address == "" then
		print("❌ Recipient address required")
		return
	end

	io.write("Enter amount in wei (or use scientific notation like 1e18 for 1 token): ")
	io.flush()
	local amount_input = io.read()

	if not amount_input or amount_input == "" then
		print("❌ Amount required")
		return
	end

	-- Convert scientific notation if needed
	local amount = amount_input
	if amount_input:match("e") then
		amount = string.format("%.0f", tonumber(amount_input))
	end

	print("\n>> Sending transaction...")
	print("From: " .. current_wallet.address)
	print("To: " .. to_address)
	print("Amount: " .. amount .. " wei")

	io.write("\nConfirm transaction? (y/n): ")
	io.flush()
	local confirm = io.read()

	if confirm and (confirm:lower() == "y" or confirm:lower() == "yes") then
		local success, tx_hash = pcall(function()
			return wallet.cronos_send_tx({
				private_key = current_wallet.private_key,
				to = to_address,
				amount = amount,
			})
		end)

		if success then
			print("\n✓ Transaction sent!")
			print("Transaction hash: " .. tx_hash)
		else
			print("❌ Error: " .. tostring(tx_hash))
		end
	else
		print("Transaction cancelled")
	end
end

-- Function to get transaction by hash
local function cronos_get_tx_menu()
	print("\n>> Get Transaction by Hash")
	print("-------------------")

	io.write("Enter transaction hash: ")
	io.flush()
	local tx_hash = io.read()

	if not tx_hash or tx_hash == "" then
		print("❌ Transaction hash required")
		return
	end

	print("\n>> Fetching transaction...")
	local success, tx = pcall(function()
		return wallet.cronos_get_tx(tx_hash)
	end)

	if success then
		print("\n>> Transaction Details:")
		print("-------------------")
		if tx.hash then
			print("Hash: " .. tx.hash)
			print("From: " .. (tx.from or "N/A"))
			print("To: " .. (tx.to or "N/A"))
			print("Value: " .. (tx.value or "0") .. " wei")
			print("Gas: " .. (tx.gas or "N/A"))
			print("Gas Price: " .. (tx.gas_price or "N/A"))
			print("Nonce: " .. (tx.nonce or "N/A"))
			print("Block: " .. (tx.block_number or "pending"))
		else
			print("Transaction not found or still pending")
		end
		print("-------------------")
	else
		print("❌ Error: " .. tostring(tx))
	end
end

-- Function to get latest block
local function cronos_get_latest_block_menu()
	print("\n>> Get Latest Block Number")
	print("-------------------")

	print("\n>> Fetching latest block...")
	local success, block_number = pcall(function()
		return wallet.cronos_get_latest_block()
	end)

	if success then
		print("\n✓ Latest Block Number: " .. block_number)
	else
		print("❌ Error: " .. tostring(block_number))
	end
end

-- Function to get ERC20 token balance
local function cronos_get_erc20_balance_menu()
	print("\n>> Get ERC20 Token Balance")
	print("-------------------")

	-- Show saved token addresses
	local saved_tokens = wallet.list_token_addresses()
	local has_saved_tokens = false
	for _ in pairs(saved_tokens) do
		has_saved_tokens = true
		break
	end

	if has_saved_tokens then
		print("\nSaved Token Addresses:")
		for name, addr in pairs(saved_tokens) do
			print("  " .. name .. ": " .. addr)
		end
		print()
	end

	io.write("Enter token contract address (or token name if saved): ")
	io.flush()
	local token_input = io.read()

	if not token_input or token_input == "" then
		print("❌ Token address required")
		return
	end

	-- Check if it's a saved token name
	local token_address = wallet.get_token_address(token_input)
	if not token_address then
		-- Not a saved name, use as direct address
		token_address = token_input
	else
		print("Using saved token: " .. token_input .. " -> " .. token_address)
	end

	local config = wallet.get_config()
	local owner_address = config.default_address

	if not owner_address then
		io.write("Enter owner address (or press Enter to use current wallet): ")
		io.flush()
		local input = io.read()
		if input and input ~= "" then
			owner_address = input
		elseif current_wallet then
			owner_address = current_wallet.address
		else
			print("❌ No owner address provided")
			return
		end
	else
		io.write("Using default address: " .. owner_address .. " (press Enter to continue or enter new address): ")
		io.flush()
		local input = io.read()
		if input and input ~= "" then
			owner_address = input
		end
	end

	print("\n>> Fetching ERC20 balance...")
	local success, balance = pcall(function()
		return wallet.cronos_get_erc20_balance({
			token_address = token_address,
			address = owner_address,
		})
	end)

	if success then
		print("\nToken: " .. token_address)
		print("Owner: " .. owner_address)
		print("Balance: " .. balance)

		-- Offer to save token address if not already saved
		if token_input == token_address then -- User entered address directly, not a saved name
			io.write("\nSave this token address for future use? (y/n): ")
			io.flush()
			local save_response = io.read()
			if save_response and (save_response:lower() == "y" or save_response:lower() == "yes") then
				io.write("Enter a name for this token (e.g., USDT, USDC): ")
				io.flush()
				local token_name = io.read()
				if token_name and token_name ~= "" then
					wallet.save_token_address(token_name, token_address)
					print("✓ Token saved as '" .. token_name .. "'")
				end
			end
		end
	else
		print("❌ Error: " .. tostring(balance))
	end
end

-- Function to send ERC20 token
local function cronos_send_erc20_menu()
	if not current_wallet then
		print("\n❌ No wallet loaded. Please create or import a wallet first.")
		return
	end

	print("\n>> Send ERC20 Token")
	print("-------------------")

	-- Show saved token addresses
	local saved_tokens = wallet.list_token_addresses()
	local has_saved_tokens = false
	for _ in pairs(saved_tokens) do
		has_saved_tokens = true
		break
	end

	if has_saved_tokens then
		print("\nSaved Token Addresses:")
		for name, addr in pairs(saved_tokens) do
			print("  " .. name .. ": " .. addr)
		end
		print()
	end

	io.write("Enter token contract address (or token name if saved): ")
	io.flush()
	local token_input = io.read()

	if not token_input or token_input == "" then
		print("❌ Token address required")
		return
	end

	-- Check if it's a saved token name
	local token_address = wallet.get_token_address(token_input)
	if not token_address then
		-- Not a saved name, use as direct address
		token_address = token_input
	else
		print("Using saved token: " .. token_input .. " -> " .. token_address)
	end

	io.write("Enter recipient address: ")
	io.flush()
	local to_address = io.read()

	if not to_address or to_address == "" then
		print("❌ Recipient address required")
		return
	end

	io.write("Enter amount (in token's smallest unit): ")
	io.flush()
	local amount_input = io.read()

	if not amount_input or amount_input == "" then
		print("❌ Amount required")
		return
	end

	-- Convert scientific notation if needed
	local amount = amount_input
	if amount_input:match("e") then
		amount = string.format("%.0f", tonumber(amount_input))
	end

	print("\n>> Sending ERC20 transaction...")
	print("Token: " .. token_address)
	print("From: " .. current_wallet.address)
	print("To: " .. to_address)
	print("Amount: " .. amount)

	io.write("\nConfirm transaction? (y/n): ")
	io.flush()
	local confirm = io.read()

	if confirm and (confirm:lower() == "y" or confirm:lower() == "yes") then
		local success, tx_hash = pcall(function()
			return wallet.cronos_send_erc20({
				private_key = current_wallet.private_key,
				token_address = token_address,
				to = to_address,
				amount = amount,
			})
		end)

		if success then
			print("\n✓ Transaction sent!")
			print("Transaction hash: " .. tx_hash)

			-- Offer to save token address if not already saved
			if token_input == token_address then -- User entered address directly, not a saved name
				io.write("\nSave this token address for future use? (y/n): ")
				io.flush()
				local save_response = io.read()
				if save_response and (save_response:lower() == "y" or save_response:lower() == "yes") then
					io.write("Enter a name for this token (e.g., USDT, USDC): ")
					io.flush()
					local token_name = io.read()
					if token_name and token_name ~= "" then
						wallet.save_token_address(token_name, token_address)
						print("✓ Token saved as '" .. token_name .. "'")
					end
				end
			end
		else
			print("❌ Error: " .. tostring(tx_hash))
		end
	else
		print("Transaction cancelled")
	end
end

-- Function to view/update config
local function cronos_config_menu()
	print("\n>> Cronos EVM Configuration")
	print("-------------------")

	local config = wallet.get_config()
	print("\nCurrent Configuration:")
	print("Chain ID: " .. config.cronos_chain_id)
	print("RPC URL: " .. config.cronos_rpc_url)
	if config.default_address then
		print("Default Address: " .. config.default_address)
	else
		print("Default Address: (not set)")
	end

	-- Show saved token addresses
	local saved_tokens = wallet.list_token_addresses()
	local token_count = 0
	for _ in pairs(saved_tokens) do
		token_count = token_count + 1
	end
	if token_count > 0 then
		print("\nSaved Token Addresses (" .. token_count .. "):")
		for name, addr in pairs(saved_tokens) do
			print("  " .. name .. ": " .. addr)
		end
	else
		print("\nSaved Token Addresses: (none)")
	end

	io.write("\nDo you want to update the configuration? (y/n): ")
	io.flush()
	local update = io.read()

	if update and (update:lower() == "y" or update:lower() == "yes") then
		io.write("Enter new chain ID (or press Enter to keep current): ")
		io.flush()
		local chain_id_input = io.read()

		io.write("Enter new RPC URL (or press Enter to keep current): ")
		io.flush()
		local rpc_url_input = io.read()

		local new_config = {}

		if chain_id_input and chain_id_input ~= "" then
			new_config.cronos_chain_id = tonumber(chain_id_input)
		else
			new_config.cronos_chain_id = config.cronos_chain_id
		end

		if rpc_url_input and rpc_url_input ~= "" then
			new_config.cronos_rpc_url = rpc_url_input
		else
			new_config.cronos_rpc_url = config.cronos_rpc_url
		end

		if config.default_address then
			new_config.default_address = config.default_address
		end

		local success, err = pcall(function()
			wallet.set_config(new_config)
		end)

		if success then
			print("\n✓ Configuration updated!")
		else
			print("❌ Error: " .. tostring(err))
		end
	end
end

-- Function to set default address
local function set_default_address_menu()
	print("\n>> Set Default Address")
	print("-------------------")
	print("\nWhat is the default address?")
	print("  The default address is used for balance checks and other")
	print("  read-only operations. It allows you to quickly check balances")
	print("  without entering an address each time.")
	print()

	local config = wallet.get_config()
	if config.default_address then
		print("Current default address: " .. config.default_address)
	else
		print("Current default address: (not set)")
	end

	-- Show saved token addresses info
	local saved_tokens = wallet.list_token_addresses()
	local token_count = 0
	for _ in pairs(saved_tokens) do
		token_count = token_count + 1
	end
	if token_count > 0 then
		print("\nYou have " .. token_count .. " saved token address(es)")
	end
	print()

	io.write("Enter new default address (or press Enter to use current wallet): ")
	io.flush()
	local address = io.read()

	if not address or address == "" then
		if current_wallet then
			address = current_wallet.address
			print("Using current wallet address: " .. address)
		else
			print("❌ No address provided")
			return
		end
	end

	config.default_address = address

	local success, err = pcall(function()
		wallet.set_config(config)
	end)

	if success then
		print("\n✓ Default address set to: " .. address)
	else
		print("❌ Error: " .. tostring(err))
	end
end

-- Function to toggle AI autorun mode
local function toggle_ai_autorun_menu()
	print("\n>> Toggle AI Autorun Mode")
	print("-------------------")
	print("\nWhat is AI Autorun?")
	print("  When enabled, AI-generated code executes automatically")
	print("  without prompting for confirmation each time.")
	print("  When disabled, you'll be asked to confirm before execution.")
	print()
	print("Current setting: " .. (ai_autorun and "ENABLED (auto-execute)" or "DISABLED (prompt before execute)"))
	print()

	io.write("Toggle autorun mode? (y/n): ")
	io.flush()
	local confirm = io.read()

	if confirm and (confirm:lower() == "y" or confirm:lower() == "yes") then
		ai_autorun = not ai_autorun
		print("\n✓ AI Autorun mode: " .. (ai_autorun and "ENABLED" or "DISABLED"))
	else
		print("\nNo changes made.")
	end
end

-- Function to pause and wait for user input
local function pause()
	io.write("\nPress Enter to continue...")
	io.flush()
	io.read()
end

-- Main program loop
local function main()
	local running = true

	-- Display wallet image if available
	local success, err = pcall(function()
		wallet.display_image("wallet.jpg")
	end)
	if not success then
		-- Show error for debugging (terminal might not support images)
		print("⚠️  Note: Could not display image - " .. tostring(err))
		print("   (Your terminal may not support image display)\n")
	end

	print("\n🔐 Welcome to ECDSA Wallet Manager")
	print("This wallet uses Lua scripting with Rust crypto backend")
	print("HD Wallet Support: BIP32/BIP44 Derivation Path\n")

	while running do
		print_menu()
		local choice = io.read()

		if choice == "1" then
			generate_new_mnemonic()
			pause()
		elseif choice == "2" then
			create_wallet_from_mnemonic()
			pause()
		elseif choice == "3" then
			show_wallet_info()
			pause()
		elseif choice == "4" then
			show_private_key()
			pause()
		elseif choice == "5" then
			show_public_key()
			pause()
		elseif choice == "6" then
			show_multiple_addresses()
			pause()
		elseif choice == "7" then
			show_address()
			pause()
		elseif choice == "8" then
			ai_mode()
			pause()
		elseif choice == "9" then
			cronos_get_balance_menu()
			pause()
		elseif choice == "10" then
			cronos_send_tx_menu()
			pause()
		elseif choice == "11" then
			cronos_get_tx_menu()
			pause()
		elseif choice == "12" then
			cronos_get_latest_block_menu()
			pause()
		elseif choice == "13" then
			cronos_get_erc20_balance_menu()
			pause()
		elseif choice == "14" then
			cronos_send_erc20_menu()
			pause()
		elseif choice == "15" then
			cronos_config_menu()
			pause()
		elseif choice == "16" then
			set_default_address_menu()
			pause()
		elseif choice == "17" then
			toggle_ai_autorun_menu()
			pause()
		elseif choice == "0" then
			print("\n👋 Thank you for using ECDSA Wallet Manager!")
			print("🔐 Remember to keep your keys safe!\n")
			running = false
		else
			print("\n❌ Invalid choice. Please try again.")
			pause()
		end

		-- Clear screen (works on Unix-like systems)
		if running then
			os.execute("clear || cls")
		end
	end
end

-- Run the main program
main()
