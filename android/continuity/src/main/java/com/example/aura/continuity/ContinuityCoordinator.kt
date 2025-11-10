package com.example.aura.continuity

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ContinuityCoordinator(
    context: Context
) {
    private val discovery = DeviceDiscovery(context)
    private val manager = ContinuityManager(context)

    suspend fun transfer(description: String): String = withContext(Dispatchers.IO) {
        val devices = discovery.nearbyDevices()
        if (devices.isEmpty()) {
            "No companion devices available for transfer: $description"
        } else {
            val target = devices.first()
            val payload = TaskTransfer(
                intent = description,
                summary = "Aura handoff: $description",
                metadata = mapOf("timestamp" to System.currentTimeMillis().toString())
            )
            val success = manager.transferTask(target, payload)
            if (success) {
                "Transferred task to $target: $description"
            } else {
                "Failed to transfer task to $target. Please try again."
            }
        }
    }
}
