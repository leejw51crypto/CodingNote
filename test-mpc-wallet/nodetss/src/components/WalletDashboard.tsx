import { Card } from "./ui/Card";
import { Button } from "./ui/Button";

export function WalletDashboard() {
  return (
    <Card className="p-6 space-y-6">
      <div className="space-y-2">
        <h2 className="text-2xl font-bold">Wallet Dashboard</h2>
        <p className="text-sm text-gray-500">Manage your TSS wallet</p>
      </div>

      <div className="space-y-4">
        <div className="p-4 rounded-lg bg-gray-100 dark:bg-gray-800">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium">Balance</span>
            <span className="text-lg font-bold">0.00 ETH</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Button variant="outline">Send</Button>
          <Button variant="outline">Receive</Button>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-medium">Recent Transactions</h3>
          <div className="text-sm text-gray-500 text-center py-8">
            No transactions yet
          </div>
        </div>
      </div>
    </Card>
  );
}
