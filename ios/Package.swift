// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "FastLLVC",
    platforms: [
        .iOS(.v17)
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
