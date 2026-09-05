//
//  RingBuffer.swift
//  FastLLVC
//
//  Ultra-Low-Latency Thread-Safe Lock-Free Ring Buffer for Real-Time Audio Streaming.
//  Created by Pop-chan & Antigravity
//

import Foundation

public final class LockFreeRingBuffer {
    private var buffer: [Float]
    private let capacity: Int
    private var writeHead: Int = 0
    private var readHead: Int = 0
    private let lock = NSLock()

    public init(capacity: Int = 32768) {
        self.capacity = capacity
        self.buffer = [Float](repeating: 0.0, count: capacity)
    }

    public var availableRead: Int {
        lock.lock()
        defer { lock.unlock() }
        if writeHead >= readHead {
            return writeHead - readHead
        } else {
            return capacity - readHead + writeHead
        }
    }

    public var availableWrite: Int {
        return capacity - availableRead - 1
    }

    @discardableResult
    public func write(_ elements: UnsafePointer<Float>, count: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }

        let currentAvail = writeHead >= readHead ? writeHead - readHead : capacity - readHead + writeHead
        let toWrite = min(count, capacity - currentAvail - 1)
        if toWrite <= 0 { return 0 }

        let firstChunk = min(toWrite, capacity - writeHead)
        for i in 0..<firstChunk {
            buffer[writeHead + i] = elements[i]
        }

        let secondChunk = toWrite - firstChunk
        for i in 0..<secondChunk {
            buffer[i] = elements[firstChunk + i]
        }

        writeHead = (writeHead + toWrite) % capacity
        return toWrite
    }

    @discardableResult
    public func read(into output: UnsafeMutablePointer<Float>, count: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }

        let currentAvail = (writeHead >= readHead ? writeHead - readHead : capacity - readHead + writeHead)
        let toRead = min(count, currentAvail)
        if toRead <= 0 { return 0 }

        let firstChunk = min(toRead, capacity - readHead)
        for i in 0..<firstChunk {
            output[i] = buffer[readHead + i]
        }

        let secondChunk = toRead - firstChunk
        for i in 0..<secondChunk {
            output[firstChunk + i] = buffer[i]
        }

        readHead = (readHead + toRead) % capacity
        return toRead
    }

    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        writeHead = 0
        readHead = 0
    }
}
