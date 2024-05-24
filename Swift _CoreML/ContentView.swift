import UIKit
import CoreML
import Vision

class test_coreml: UIViewController {

    var model: VNCoreMLModel?
    var imageView: UIImageView!
    var overlayView: UIView!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupViews()
        loadModel()
        performPrediction()
    }
    
    private func setupViews() {
        // UIImageViewの設定
        imageView = UIImageView(frame: self.view.bounds)
        imageView.contentMode = .scaleAspectFit
        self.view.addSubview(imageView)
        
        // オーバーレイビューの設定
        overlayView = UIView(frame: self.view.bounds)
        overlayView.backgroundColor = .clear
        self.view.addSubview(overlayView)
    }
    
    private func loadModel() {
        // mlpackageのロード
        guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlpackage"),
              let compiledModelURL = try? MLModel.compileModel(at: modelURL),
              let mlModel = try? MLModel(contentsOf: compiledModelURL) else {
            print("Failed to load the model.")
            return
        }
        
        // Vision フレームワークを使用してモデルをロード
        do {
            let visionModel = try VNCoreMLModel(for: mlModel)
            self.model = visionModel
        } catch {
            print("Failed to create VNCoreMLModel: \(error)")
        }
    }
    
    private func performPrediction() {
        // テストデータを使用して予測を行う例
        if let model = self.model {
            // 例えば、画像データを読み込み、予測を行う
            guard let image = UIImage(named: "car") else {
                print("Failed to load the image.")
                return
            }
            
            imageView.image = image
            
            guard let buffer = image.toBuffer() else {
                print("Failed to convert image to buffer.")
                return
            }
            
            // 入力画像からバッファを作成
            let request = VNCoreMLRequest(model: model) { [weak self] (request, error) in
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    DispatchQueue.main.async {
                        self?.overlayView.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
                        for result in results {
                            self?.drawBoundingBox(boundingBox: result.boundingBox, confidence: result.confidence)
                        }
                    }
                } else if let error = error {
                    print("Prediction failed: \(error)")
                }
            }
            
            // リクエストの設定
            request.imageCropAndScaleOption = .centerCrop
            
            let handler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("Failed to perform request: \(error)")
            }
        }
    }
    
    private func drawBoundingBox(boundingBox: CGRect, confidence: VNConfidence) {
        let width = overlayView.frame.width
        let height = overlayView.frame.height
        
        // バウンディングボックスの位置とサイズを計算
        let rect = CGRect(
            x: boundingBox.origin.x * width,
            y: (1 - boundingBox.origin.y - boundingBox.height) * height,
            width: boundingBox.width * width,
            height: boundingBox.height * height
        )
        
        // バウンディングボックスのレイヤーを作成
        let boxLayer = CALayer()
        boxLayer.frame = rect
        boxLayer.borderColor = UIColor.red.cgColor
        boxLayer.borderWidth = 2.0
        overlayView.layer.addSublayer(boxLayer)
        
        // 信頼度のラベルを追加
        let textLayer = CATextLayer()
        textLayer.string = String(format: "Conf: %.2f", confidence)
        textLayer.fontSize = 14
        textLayer.foregroundColor = UIColor.red.cgColor
        textLayer.frame = CGRect(x: rect.origin.x, y: rect.origin.y - 20, width: 100, height: 20)
        overlayView.layer.addSublayer(textLayer)
    }
}

// UIImageをバッファに変換するためのヘルパー関数
extension UIImage {
    func toBuffer() -> CVPixelBuffer? {
        let image = self.cgImage!
        let options: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        
        var pxbuffer: CVPixelBuffer?
        let width = image.width
        let height = image.height
        let attrs = options as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &pxbuffer)
        
        guard let buffer = pxbuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let pxdata = CVPixelBufferGetBaseAddress(buffer)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pxdata, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}

