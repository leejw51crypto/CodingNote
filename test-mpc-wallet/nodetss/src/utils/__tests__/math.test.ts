import { add } from "../math";

describe("add function", () => {
  test("adds two positive numbers correctly", () => {
    expect(add(1, 2)).toBe(3);
  });

  test("adds a positive and a negative number correctly", () => {
    expect(add(5, -3)).toBe(2);
  });

  test("adds two negative numbers correctly", () => {
    expect(add(-1, -2)).toBe(-3);
  });

  test("adds zero correctly", () => {
    expect(add(0, 5)).toBe(5);
    expect(add(5, 0)).toBe(5);
    expect(add(0, 0)).toBe(0);
  });
});
