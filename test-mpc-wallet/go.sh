source .env

python tssjson.py

cd rusttss
cargo test -- --nocapture
cd ..

cd nodetss
yarn test
cd ..


