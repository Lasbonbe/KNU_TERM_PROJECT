#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace cv::dnn;

int main()
{
    cv::Mat image = cv::imread("./images_for_test/tank1.jpg");  // Remplace par le chemin de l'image à tester
    if (image.empty()) {
        std::cerr << "Erreur lors du chargement de l'image." << std::endl;
        return -1;
    }
    else std::cout << "Image chargée avec succès." << std::endl;

    std::string modelPath = "models/datasetV2/saved_model.pb";  // Remplace par le chemin de ton modèle
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelPath);



    // Vérifier que le modèle est chargé correctement
    if (net.empty()) {
        std::cerr << "Erreur de chargement du modèle." << std::endl;
        return -1;
    }




    // Prétraiter l'image pour le modèle (redimensionner et normaliser)
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(224, 224), cv::Scalar(104, 177, 123), true, false);

    // Passer l'image dans le réseau
    net.setInput(blob);
    cv::Mat output = net.forward();

    // Afficher les résultats (par exemple, les scores de classification)
    std::cout << "Résultat de l'inférence : " << output << std::endl;

    // Si ton modèle fait de la classification, tu peux obtenir la classe avec la valeur maximale :
    cv::Point classId;
    double confidence;
    cv::minMaxLoc(output, 0, &confidence, 0, &classId);
    std::cout << "Classe prédite : " << classId.x << " avec une confiance de : " << confidence << std::endl;

    return 0;

}