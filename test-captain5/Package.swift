// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "BookExample",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(name: "BookExample", targets: ["BookExample"])
    ],
    targets: [
        .systemLibrary(
            name: "CCapnp",
            pkgConfig: "capnp",
            providers: [
                .brew(["capnp"])
            ]
        ),
        .executableTarget(
            name: "BookExample",
            dependencies: ["CCapnp"],
            cxxSettings: [
                .headerSearchPath("../../build"),
                .headerSearchPath("Sources/CCapnp/include")
            ],
            linkerSettings: [
                .linkedLibrary("book_wrapper"),
                .linkedLibrary("capnp"),
                .linkedLibrary("kj"),
                .unsafeFlags(["-L../../build/lib"])
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
) 