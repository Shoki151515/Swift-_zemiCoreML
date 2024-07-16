import UIKit
import CoreML
import Vision
import AVFoundation

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var detectionOverlay: CALayer! = nil
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }
    
    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo
        
        guard let backCamera = AVCaptureDevice.default(for: AVMediaType.video) else {
            print("Unable to access back camera!")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: backCamera)
            captureSession.addInput(input)
            
            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            captureSession.addOutput(output)
            
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewLayer.videoGravity = .resizeAspectFill
            previewLayer.frame = view.frame
            view.layer.addSublayer(previewLayer)
            
            captureSession.startRunning()
        } catch let error  {
            print("Error Unable to initialize back camera:  \(error.localizedDescription)")
        }
        
        detectionOverlay = CALayer()
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: view.bounds.width,
                                         height: view.bounds.height)
        detectionOverlay.position = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
        view.layer.addSublayer(detectionOverlay)
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let requestOptions: [VNImageOption: Any] = [:]
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: requestOptions)
        
        do {
            try imageRequestHandler.perform([self.detectionRequest])
        } catch {
            print(error)
        }
    }
    
    lazy var detectionRequest: VNCoreMLRequest = {
        do {
            guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") else {
                fatalError("Model file not found.")
            }
            let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            return VNCoreMLRequest(model: model, completionHandler: self.handleDetection)
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    func handleDetection(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            return
        }

        DispatchQueue.main.async {
            self.detectionOverlay.sublayers?.removeSubrange(0...)

            for observation in observations {
                let boundingBox = observation.boundingBox
                let transformedBoundingBox = self.transformBoundingBox(boundingBox)
                let boundingBoxPath = CGPath(rect: transformedBoundingBox, transform: nil)

                let shapeLayer = CAShapeLayer()
                shapeLayer.path = boundingBoxPath
                shapeLayer.strokeColor = UIColor.red.cgColor
                shapeLayer.fillColor = UIColor.clear.cgColor
                shapeLayer.lineWidth = 2

                self.detectionOverlay.addSublayer(shapeLayer)

                if let topLabelObservation = observation.labels.first {
                    let textLayer = CATextLayer()
                    textLayer.string = "\(topLabelObservation.identifier) \(String(format: "%.2f", topLabelObservation.confidence * 100))%"
                    textLayer.foregroundColor = UIColor.white.cgColor
                    textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
                    textLayer.fontSize = 14
                    let labelWidth: CGFloat = 300
                    let labelHeight: CGFloat = 20
                    textLayer.frame = CGRect(x: transformedBoundingBox.origin.x, y: transformedBoundingBox.origin.y - labelHeight, width: labelWidth, height: labelHeight)
                    textLayer.alignmentMode = .center
                    textLayer.contentsScale = UIScreen.main.scale

                    self.detectionOverlay.addSublayer(textLayer)
                }
                
                // OCRを実行する
                self.performOCR(on: transformedBoundingBox)
            }
        }
    }

    func transformBoundingBox(_ boundingBox: CGRect) -> CGRect {
        let x = boundingBox.origin.x * view.bounds.width
        let y = (1 - boundingBox.origin.y - boundingBox.height) * view.bounds.height

        let widthFactor: CGFloat = 1.5
        let heightFactor: CGFloat = 1.0

        let width = boundingBox.width * view.bounds.width * widthFactor
        let height = boundingBox.height * view.bounds.height * heightFactor

        let adjustedX = x - (width - boundingBox.width * view.bounds.width) / 2
        let adjustedY = y - (height - boundingBox.height * view.bounds.height) / 2

        return CGRect(x: adjustedX, y: adjustedY, width: width, height: height)
    }
    
    func performOCR(on boundingBox: CGRect) {
        let image = captureImageFromPreviewLayer()
        guard let cgImage = image?.cgImage?.cropping(to: boundingBox) else {
            return
        }
        
        recognize(cgImage: cgImage) { texts in
            DispatchQueue.main.async {
                for (index, text) in texts.enumerated() {
                    let textLayer = CATextLayer()
                    textLayer.string = text
                    textLayer.foregroundColor = UIColor.white.cgColor
                    textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
                    textLayer.fontSize = 14
                    let labelWidth: CGFloat = 300
                    let labelHeight: CGFloat = 20
                    textLayer.frame = CGRect(x: boundingBox.origin.x, y: boundingBox.origin.y + CGFloat(index * Int(labelHeight)), width: labelWidth, height: labelHeight)
                    textLayer.alignmentMode = .center
                    textLayer.contentsScale = UIScreen.main.scale

                    self.detectionOverlay.addSublayer(textLayer)
                }
            }
        }
    }
    
    func captureImageFromPreviewLayer() -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(view.bounds.size, false, 0.0)
        view.drawHierarchy(in: view.bounds, afterScreenUpdates: true)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
    
    func recognize(cgImage: CGImage, handler: @escaping ([String]) -> Void) {
        var texts: [String] = []
        let request = VNRecognizeTextRequest { (request, error) in
            guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
            for observation in observations {
                let candidates = observation.topCandidates(5)
                for candidate in candidates {
                    print(candidate.string)
                }
                texts.append(candidates.first!.string)
            }
            handler(texts)
        }

        request.recognitionLanguages = ["ja-JP"]
        let handler = VNImageRequestHandler(cgImage: cgImage)
        try? handler.perform([request])
    }
}

