"use client";

import dynamic from "next/dynamic";

const TSSWallet = dynamic(
  () => import("./TSSWallet").then((mod) => mod.TSSWallet),
  {
    ssr: false,
  },
);

export default function ClientTSSWallet() {
  return <TSSWallet />;
}
