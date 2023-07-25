all: prepare
	npx hardhat compile
	npx hardhat run scripts/balance.ts --network my
	npx hardhat run scripts/swap.ts --network my
	npx hardhat run scripts/balance.ts --network my

time:
	npx hardhat run scripts/time.ts --network my
prepare:
	yarn
ok:
	npx hardhat node

format:
	npm run format

install:
	npm install
