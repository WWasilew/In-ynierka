package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 25f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 25f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            canvas.drawRect(left, top, right, bottom, boxPaint)
            val drawableText = it.clsName

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)

        }
    }

    fun drawBoxesOnBitmap(bitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)
        val paint = Paint().apply {
            color = boxPaint.color
            style = Paint.Style.STROKE
            strokeWidth = boxPaint.strokeWidth
        }
        val textPaintLocal = Paint(textPaint)
        val textBgPaintLocal = Paint(textBackgroundPaint)
        val bounds = Rect()

        boxes.forEach {
            val left = it.x1 * bitmap.width
            val top = it.y1 * bitmap.height
            val right = it.x2 * bitmap.width
            val bottom = it.y2 * bitmap.height

            canvas.drawRect(left, top, right, bottom, paint)

            val drawableText = it.clsName
            textBgPaintLocal.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()

            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBgPaintLocal
            )
            canvas.drawText(drawableText, left, top + bounds.height(), textPaintLocal)
        }
        return mutable
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}