//import UIKit
//import CoreML
//import Vision
//import AVFoundation
//
//
//func recognize(cgImage: CGImage, handler: @escaping([String]) -> Void) {
//    var texts: [String] = []
//    let request = VNRecognizeTextRequest { (request, error) in
//        guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
//        for observation in observations {
//            let candidates = observation.topCandidates(5)
//            for candidate in candidates {
//                print(candidate.string)
//            }
//            texts.append(candidates.first!.string)
//        }
//        handler(texts)
//    }
//
//    request.recognitionLanguages = ["ja-JP"]
//    let handler = VNImageRequestHandler(cgImage: cgImage)
//    try? handler.perform([request])
//}
