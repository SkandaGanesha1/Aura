package com.example.aura.perception

import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo

/**
 * Traverses the AccessibilityNodeInfo tree and extracts useful metadata.
 */
object AccessibilityParser {

    fun describe(node: AccessibilityNodeInfo?): List<UiElement> {
        if (node == null) return emptyList()
        val elements = mutableListOf<UiElement>()
        traverse(node, elements, depth = 0)
        Log.d(TAG, "Parsed ${elements.size} elements from accessibility tree")
        return elements
    }

    private fun traverse(node: AccessibilityNodeInfo, sink: MutableList<UiElement>, depth: Int) {
        val element = UiElement(
            text = node.text?.toString().orEmpty(),
            contentDescription = node.contentDescription?.toString().orEmpty(),
            className = node.className?.toString().orEmpty(),
            bounds = node.boundsToShortString(),
            depth = depth
        )
        sink += element
        for (i in 0 until node.childCount) {
            traverse(node.getChild(i), sink, depth + 1)
        }
    }

    private const val TAG = "AccessibilityParser"
}

data class UiElement(
    val text: String,
    val contentDescription: String,
    val className: String,
    val bounds: String,
    val depth: Int
)

private fun AccessibilityNodeInfo.boundsToShortString(): String {
    val rect = Rect()
    getBoundsInScreen(rect)
    return rect.toShortString()
}
