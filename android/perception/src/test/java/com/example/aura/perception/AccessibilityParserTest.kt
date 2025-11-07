package com.example.aura.perception

import android.view.accessibility.AccessibilityNodeInfo
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.mockito.Mockito
import org.mockito.Mockito.mock

class AccessibilityParserTest {

    @Test
    fun describe_skipsNullChildrenAndRecyclesNonNullNodes() {
        val root = mock(AccessibilityNodeInfo::class.java)
        val child = mock(AccessibilityNodeInfo::class.java)

        Mockito.`when`(root.text).thenReturn("root")
        Mockito.`when`(root.contentDescription).thenReturn(null)
        Mockito.`when`(root.className).thenReturn("RootClass")
        Mockito.`when`(root.childCount).thenReturn(2)
        Mockito.`when`(root.getChild(0)).thenReturn(child)
        Mockito.`when`(root.getChild(1)).thenReturn(null)

        Mockito.`when`(child.text).thenReturn("leaf")
        Mockito.`when`(child.contentDescription).thenReturn("leaf-desc")
        Mockito.`when`(child.className).thenReturn("LeafClass")
        Mockito.`when`(child.childCount).thenReturn(0)

        val elements = AccessibilityParser.describe(root)

        assertThat(elements).hasSize(2)
        assertThat(elements[0].text).isEqualTo("root")
        assertThat(elements[1].text).isEqualTo("leaf")
        assertThat(elements[1].depth).isEqualTo(1)

        Mockito.verify(child).recycle()
        Mockito.verify(root, Mockito.never()).recycle()
    }
}
