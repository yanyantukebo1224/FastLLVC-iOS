// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FastLLVC",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "FastLLVCCore",
            targets: ["FastLLVCCore"]
        ),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "FastLLVCCore",
            dependencies: [],
            path: "FastLLVC",
            exclude: ["App/Info.plist", "Assets.xcassets"]
        ),
    ]
)
