# HTTP/3 Test Server with Chrome

This project demonstrates an HTTP/3 server implementation that supports both HTTP/1.1 (TCP) and HTTP/3 (QUIC/UDP) protocols.

## Certificate Setup

1. Generate certificates using `makecert.sh`
2. Add the generated `cert.pem` to Keychain Access for Safari compatibility
3. The server will be accessible at:
   - HTTPS (HTTP/1.1): `https://localhost:4433` (TCP)
   - HTTP/3: `https://localhost:4433` (UDP)

## Implementation Options

### 1. Quiche (by Cloudflare)
- Written in C with Rust bindings
- Production-ready implementation used by Cloudflare
- Provides both client and server implementations
- More mature and battle-tested in production environments

### 2. Quinn (Rust)
- Pure Rust implementation
- Good for Rust-only projects
- Active community development
- Excellent for learning and understanding QUIC/HTTP/3

## Protocol Details

HTTP/3 negotiation requires initial HTTP/1.1 support for the ALPN (Application-Layer Protocol Negotiation) process. The server handles both:
- TCP connections for HTTP/1.1
- UDP connections for HTTP/3 (QUIC)

## Security Notes

- Always use HTTPS for both HTTP/1.1 and HTTP/3
- Certificates are required for secure communication
- Local development certificates should not be used in production

## Getting Started

1. Run `makecert.sh` to generate certificates
2. Add `cert.pem` to your system's keychain
3. Start the server (implementation specific instructions to be added)
4. Access the server through Chrome or Safari at https://localhost:4433
