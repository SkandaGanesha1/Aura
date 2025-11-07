package com.example.aura.desktop

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class ContinuityServer {

    private val scope = CoroutineScope(Dispatchers.Default)
    private var heartbeatJob: Job? = null

    fun start() {
        if (heartbeatJob != null) return
        heartbeatJob = scope.launch {
            while (true) {
                delay(5_000)
                println("Continuity heartbeat… (no sessions connected)")
            }
        }
    }

    fun awaitTermination() {
        Runtime.getRuntime().addShutdownHook(Thread {
            scope.cancel()
        })
        while (true) {
            Thread.sleep(1_000)
        }
    }
}
