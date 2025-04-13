import * as $ from "capnp-es";
export declare const _capnpFileId: bigint;
export declare class Book extends $.Struct {
  static readonly _capnp: {
    displayName: string;
    id: string;
    size: $.ObjectSize;
  };
  get title(): string;
  set title(value: string);
  get author(): string;
  set author(value: string);
  get pages(): number;
  set pages(value: number);
  get publishYear(): number;
  set publishYear(value: number);
  _adoptGenres(value: $.Orphan<$.List<string>>): void;
  _disownGenres(): $.Orphan<$.List<string>>;
  get genres(): $.List<string>;
  _hasGenres(): boolean;
  _initGenres(length: number): $.List<string>;
  set genres(value: $.List<string>);
  get isAvailable(): boolean;
  set isAvailable(value: boolean);
  toString(): string;
}
export declare class Fruit extends $.Struct {
  static readonly _capnp: {
    displayName: string;
    id: string;
    size: $.ObjectSize;
  };
  get name(): string;
  set name(value: string);
  get color(): string;
  set color(value: string);
  get weightGrams(): number;
  set weightGrams(value: number);
  get isRipe(): boolean;
  set isRipe(value: boolean);
  get variety(): string;
  set variety(value: string);
  toString(): string;
}
export declare const Message_Which: {
  readonly BOOK: 0;
  readonly FRUIT: 1;
};
export type Message_Which = (typeof Message_Which)[keyof typeof Message_Which];
export declare class Message extends $.Struct {
  static readonly BOOK: 0;
  static readonly FRUIT: 1;
  static readonly _capnp: {
    displayName: string;
    id: string;
    size: $.ObjectSize;
  };
  _adoptBook(value: $.Orphan<Book>): void;
  _disownBook(): $.Orphan<Book>;
  get book(): Book;
  _hasBook(): boolean;
  _initBook(): Book;
  get _isBook(): boolean;
  set book(value: Book);
  _adoptFruit(value: $.Orphan<Fruit>): void;
  _disownFruit(): $.Orphan<Fruit>;
  get fruit(): Fruit;
  _hasFruit(): boolean;
  _initFruit(): Fruit;
  get _isFruit(): boolean;
  set fruit(value: Fruit);
  toString(): string;
  which(): Message_Which;
}
export declare class BookList extends $.Struct {
  static readonly _capnp: {
    displayName: string;
    id: string;
    size: $.ObjectSize;
  };
  static _Books: $.ListCtor<Book>;
  _adoptBooks(value: $.Orphan<$.List<Book>>): void;
  _disownBooks(): $.Orphan<$.List<Book>>;
  get books(): $.List<Book>;
  _hasBooks(): boolean;
  _initBooks(length: number): $.List<Book>;
  set books(value: $.List<Book>);
  toString(): string;
}
