all: install
	npx hardhat run ./scripts/hello.ts --network cronosZkEvmTestnet

install:
	npm i

pipinstall:
	pip install zksync2==1.2.0 web3==6.10.0 eth-account==0.5.9 --no-deps
	pip install py-solc-x==2.0.3
	pip install eth-abi==2.2.0 eth-hash==0.3.3 eth-typing==2.3.0 eth-utils==1.10.0 --no-deps
	@echo "Python 3.10.14"

python:
	python ./scripts/compile.py
	python ./scripts/hello.py

wallet:
	npx hardhat run ./scripts/wallet.ts --network cronosZkEvmTestnet

compile:
	npx hardhat compile
