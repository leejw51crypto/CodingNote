import ClientTSSWallet from "../components/ClientTSSWallet";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-100">
      <div className="container mx-auto py-8">
        <ClientTSSWallet />
      </div>
    </main>
  );
}
