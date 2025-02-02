import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function shortenAddress(address: string): string {
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

export function formatBalance(balance: bigint, decimals: number = 18): string {
  const divisor = BigInt(10) ** BigInt(decimals);
  const integerPart = balance / divisor;
  const fractionalPart = balance % divisor;

  return `${integerPart}.${fractionalPart.toString().padStart(decimals, "0")}`;
}
